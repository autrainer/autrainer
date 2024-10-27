from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch

from autrainer.core.constants import TrainingConstants
from autrainer.transforms import SmartCompose

from .abstract_dataset import AbstractDataset
from .utils import (
    AbstractTargetTransform,
    LabelEncoder,
    MinMaxScaler,
    MultiLabelEncoder,
    MultiTargetMinMaxScaler,
)


class ToyDatasetWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Union[list, str],
        feature_shape: Union[int, List[int]],
        dtype: str,
        generator: torch.Generator,
        transform: SmartCompose = None,
        target_transform: SmartCompose = None,
    ) -> None:
        self.df = df
        self.target_column = target_column
        self.feature_shape = feature_shape
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.generator = generator

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        target = self.df.iloc[index][self.target_column]
        if isinstance(target, pd.Series):
            target = target.values
        if self.dtype == "float32":
            data = torch.rand(
                self.feature_shape,
                dtype=torch.float32,
                generator=self.generator,
            )
        else:
            data = torch.randint(
                0,
                255,
                self.feature_shape,
                dtype=torch.uint8,
                generator=self.generator,
            )
        if self.transform:
            img = self.transform(data, index=index)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target, index


class ToyDataset(AbstractDataset):
    def __init__(
        self,
        task: str,
        size: int,
        num_targets: int,
        feature_shape: Union[int, List[int]],
        dev_split: float,
        test_split: float,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        batch_size: int,
        dtype: str = "float32",
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
    ) -> None:
        """Toy dataset for testing purposes.

        Args:
            task: Task of the dataset in ["classification", "regression",
                "ml-classification", "mt-regression"].
            size: Size of the dataset.
            num_targets: Number of targets.
            feature_shape: Shape of the features.
            dev_split: Proportion of the dataset to use for the development set.
            test_split: Proportion of the dataset to use for the test set.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
        """
        if task not in TrainingConstants().TASKS:
            raise ValueError(
                f"Invalid task '{task}', "
                f"must be in {TrainingConstants().TASKS}."
            )
        self.size = size
        self.num_targets = num_targets
        self.feature_shape = feature_shape
        self._assert_splits(dev_split, test_split)
        self.dev_split = dev_split
        self.test_split = test_split
        self._generator = torch.Generator().manual_seed(seed)
        if dtype not in ["float32", "uint8"]:
            raise ValueError(
                f"Invalid dtype='{dtype}', must be in ['float32', 'uint8']."
            )
        self.dtype = dtype
        super().__init__(
            task=task,
            path="/",
            features_subdir="",
            file_type="",
            index_column="index",
            target_column="",
            file_handler="autrainer.datasets.utils.IdentityFileHandler",
            seed=seed,
            metrics=metrics,
            tracking_metric=tracking_metric,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=None,
        )

    @staticmethod
    def _assert_splits(dev_split: int, test_split: float) -> None:
        if dev_split + test_split >= 1.0:
            raise ValueError(
                f"Sum of dev_split '{dev_split}' and test_split "
                f"'{test_split}' must be less than 1."
            )

    def _get_dataset_sizes(self) -> Tuple[int, int, int]:
        if self.size <= 0:
            raise ValueError(f"Size must be > 0, got '{self.size}'.")

        dev_size = int(self.size * self.dev_split)
        test_size = int(self.size * self.test_split)
        train_size = self.size - dev_size - test_size
        if any(size <= 0 for size in (train_size, dev_size, test_size)):
            raise ValueError(
                f"All subsets must be > 0, got train_size '{train_size}', "
                f"dev_size '{dev_size}', test_size '{test_size}'."
            )
        return train_size, dev_size, test_size

    def load_dataframes(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.seed)

        def _reset_index(df: pd.DataFrame) -> pd.DataFrame:
            return df.reset_index(drop=True).reset_index(
                names=[self.index_column]
            )

        if self.task == "ml-classification":
            self.target_column = [
                f"class_{i+1}" for i in range(self.num_targets)
            ]
            df = pd.DataFrame(
                {
                    **{
                        col: rng.integers(0, 2, size=self.size).astype(
                            np.float32
                        )
                        for col in self.target_column
                    }
                }
            )
        elif self.task == "classification":
            self.target_column = "target"
            df = pd.DataFrame(
                {
                    self.target_column: rng.integers(
                        0, self.num_targets, size=self.size
                    )
                }
            )
            df[self.target_column] = df[self.target_column].apply(
                lambda x: f"class_{x+1}"
            )
        elif self.task == "mt-regression":
            self.target_column = [
                f"target_{i+1}" for i in range(self.num_targets)
            ]
            df = pd.DataFrame(
                {
                    **{
                        col: rng.random((self.size,)).astype(np.float32)
                        for col in self.target_column
                    }
                }
            )
        elif self.task == "regression":
            self.target_column = "target"
            df = pd.DataFrame({self.target_column: rng.random((self.size,))})
        else:
            raise ValueError(
                f"Invalid task '{self.task}', "
                f"must be in {TrainingConstants().TASKS}."
            )

        train_size, dev_size, _ = self._get_dataset_sizes()
        return (
            _reset_index(df.iloc[:train_size]),
            _reset_index(df.iloc[train_size : train_size + dev_size]),
            _reset_index(df.iloc[train_size + dev_size :]),
        )

    def _init_dataset(
        self,
        df: pd.DataFrame,
        transform: SmartCompose,
    ) -> ToyDatasetWrapper:
        return ToyDatasetWrapper(
            df=df,
            target_column=self.target_column,
            feature_shape=self.feature_shape,
            dtype=self.dtype,
            generator=self._generator,
            transform=transform,
            target_transform=self.target_transform,
        )

    @cached_property
    def target_transform(self) -> AbstractTargetTransform:
        if self.task == "ml-classification":
            return MultiLabelEncoder(0.5, self.target_column)
        elif self.task == "classification":
            return LabelEncoder(
                self.df_train[self.target_column].unique().tolist()
            )
        elif self.task == "mt-regression":
            return MultiTargetMinMaxScaler(
                target=self.target_column,
                minimum=self.df_train[self.target_column].min().to_list(),
                maximum=self.df_train[self.target_column].max().to_list(),
            )
        elif self.task == "regression":
            return MinMaxScaler(
                target=self.target_column,
                minimum=self.df_train[self.target_column].min(),
                maximum=self.df_train[self.target_column].max(),
            )
        else:
            raise ValueError(
                f"Invalid task '{self.task}', "
                f"must be in {TrainingConstants().TASKS}."
            )

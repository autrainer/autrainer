from abc import ABC, abstractmethod
from functools import cached_property
import os
from typing import Dict, List, Optional, Tuple, TypeVar, Union

from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import DataLoader

import autrainer
from autrainer.core.constants import TrainingConstants
from autrainer.metrics import AbstractMetric
from autrainer.transforms import SmartCompose

from .utils import (
    AbstractFileHandler,
    AbstractTargetTransform,
    DatasetWrapper,
    LabelEncoder,
    MinMaxScaler,
    MultiLabelEncoder,
    MultiTargetMinMaxScaler,
)


T = TypeVar("T")


class AbstractDataset(ABC):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        task: str,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: Union[str, List[str]],
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        batch_size: int,
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
    ) -> None:
        """Abstract dataset class.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
            seed: Seed for reproducibility.
            task: Task of the dataset in
                :const:`~autrainer.core.constants.TrainingConstants.TASKS`.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
        """
        self._assert_task(task)
        self._assert_directory(path, features_subdir)
        self.path = path
        self.features_subdir = features_subdir
        self.seed = seed
        self.task = task
        self.metrics = [self._init_metric(m) for m in metrics]
        self.tracking_metric = self._init_metric(tracking_metric)
        self.index_column = index_column
        self.target_column = target_column
        self.file_type = file_type
        self.file_handler = self._init_file_handler(file_handler)
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size or batch_size
        self.train_transform = train_transform or SmartCompose([])
        self.dev_transform = dev_transform or SmartCompose([])
        self.test_transform = test_transform or SmartCompose([])
        self.stratify = stratify or []

        self._generator = torch.Generator().manual_seed(self.seed)
        self.df_train, self.df_dev, self.df_test = self.load_dataframes()
        self._assert_stratify()

    @staticmethod
    def _assert_task(task: str) -> None:
        if task not in TrainingConstants().TASKS:
            raise ValueError(f"Task '{task}' not supported.")

    @staticmethod
    def _assert_directory(path: str, features_subdir: str) -> None:
        if not os.path.isdir(path):
            raise ValueError(f"Directory path='{path}' does not exist.")
        if not os.path.isdir(os.path.join(path, features_subdir)):
            raise ValueError(
                f"Directory features_subdir='{features_subdir}' "
                f"does not exist in '{path}'."
            )

    def _assert_stratify(self) -> None:
        if self.task == "ml-classification" and self.stratify:
            raise ValueError(
                "Stratify is not supported for 'ml-classification' tasks."
            )

        for column in self.stratify:
            if (
                column not in self.df_dev.columns
                or column not in self.df_test.columns
            ):
                raise ValueError(
                    f"Stratify column '{column}' is not present in "
                    "the development or test dataframes."
                )

    @property
    @abstractmethod
    def target_transform(self) -> AbstractTargetTransform:
        """Get the transform to apply to the target.

        Returns:
            Target transform.
        """

    @cached_property
    def output_dim(self) -> int:
        """Get the output dimension of the dataset.

        Returns:
            Output dimension.
        """
        return len(self.target_transform)

    @staticmethod
    def _init_metric(metric: Union[str, DictConfig, Dict]) -> AbstractMetric:
        return autrainer.instantiate_shorthand(
            config=metric,
            instance_of=AbstractMetric,
        )

    @staticmethod
    def _init_file_handler(
        file_handler: Union[str, DictConfig, Dict],
    ) -> AbstractFileHandler:
        return autrainer.instantiate_shorthand(
            config=file_handler,
            instance_of=AbstractFileHandler,
        )

    def load_dataframes(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the dataframes.

        Returns:
            Dataframes for training, development, and testing.
        """
        return (
            pd.read_csv(os.path.join(self.path, "train.csv")),
            pd.read_csv(os.path.join(self.path, "dev.csv")),
            pd.read_csv(os.path.join(self.path, "test.csv")),
        )

    def _init_dataset(
        self,
        df: pd.DataFrame,
        transform: SmartCompose,
    ) -> DatasetWrapper:
        """Initialize a wrapper around torch.utils.data.Dataset.

        Args:
            df: Dataframe to use.
            transform: Transform to apply to the features.

        Returns:
            Initialized dataset.
        """
        return DatasetWrapper(
            path=self.path,
            features_subdir=self.features_subdir,
            index_column=self.index_column,
            target_column=self.target_column,
            file_type=self.file_type,
            file_handler=self.file_handler,
            df=df,
            transform=transform,
            target_transform=self.target_transform,
        )

    @cached_property
    def train_dataset(self) -> DatasetWrapper:
        """Get the training dataset.

        Returns:
            Training dataset.
        """
        return self._init_dataset(self.df_train, self.train_transform)

    @cached_property
    def dev_dataset(self) -> DatasetWrapper:
        """Get the development dataset.

        Returns:
            Development dataset.
        """
        return self._init_dataset(self.df_dev, self.dev_transform)

    @cached_property
    def test_dataset(self) -> DatasetWrapper:
        """Get the test dataset.

        Returns:
            Test dataset.
        """
        return self._init_dataset(self.df_test, self.test_transform)

    @cached_property
    def train_loader(self) -> DataLoader:
        """Get the training loader.

        Returns:
            Training loader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._generator,
            collate_fn=self.train_transform.get_collate_fn(self),
        )

    @cached_property
    def dev_loader(self) -> DataLoader:
        """Get the development loader.

        Returns:
            Development loader.
        """
        return DataLoader(
            self.dev_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            generator=self._generator,
            collate_fn=self.dev_transform.get_collate_fn(self),
        )

    @cached_property
    def test_loader(self) -> DataLoader:
        """Get the test loader.

        Returns:
            Test loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            generator=self._generator,
            collate_fn=self.dev_transform.get_collate_fn(self),
        )

    def get_evaluation_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], AbstractTargetTransform]:
        """Get the evaluation data.

        Returns:
            Dataframes for development and testing, columns to stratify on, and
                the target transform.
        """
        return self.df_dev, self.df_test, self.stratify, self.target_transform

    @staticmethod
    def download(path: str) -> None:
        """Download the dataset. Can be implemented by subclasses, but is not
        required.

        Args:
            path: Path to download the dataset to.
        """


class BaseClassificationDataset(AbstractDataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: str,
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        batch_size: int,
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
    ) -> None:
        """Base classification dataset.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
        """
        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            task="classification",
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )

    @cached_property
    def target_transform(self) -> LabelEncoder:
        return LabelEncoder(
            self.df_train[self.target_column].unique().tolist()
        )

    @staticmethod
    def _assert_dev_split(dev_split: float) -> None:
        if not 0 <= dev_split < 1:
            raise ValueError(f"dev_split '{dev_split}' must be in [0, 1).")

    @staticmethod
    def _assert_choice(choice: T, choices: List[T], name: str) -> None:
        if choice not in choices:
            raise ValueError(f"{name} '{choice}' not in {choices}.")


class BaseMLClassificationDataset(AbstractDataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: List[str],
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        batch_size: int,
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        """Base multi-label classification dataset.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
            threshold: Threshold for classification. Defaults to 0.5.
        """
        self._assert_threshold(threshold)
        self.threshold = threshold
        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            task="ml-classification",
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )
        self._assert_target_column(allowed_columns=self.df_train.columns)

    @staticmethod
    def _assert_threshold(threshold: float) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError(
                f"Threshold '{threshold}' must be between 0 and 1."
            )

    def _assert_target_column(self, allowed_columns: List[str]) -> None:
        if isinstance(self.target_column, str):
            raise ValueError(
                f"Invalid target column '{self.target_column}'. "
                "Must be a list of strings for multi-label classification."
            )
        if not self.target_column:
            raise ValueError("No target columns provided.")

        if not all(c in allowed_columns for c in self.target_column):
            raise ValueError(
                f"Invalid target columns '{self.target_column}'. "
                f"Must be in {allowed_columns}."
            )

    @cached_property
    def target_transform(self) -> MultiLabelEncoder:
        return MultiLabelEncoder(self.threshold, self.target_column)


class BaseRegressionDataset(AbstractDataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: str,
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        batch_size: int,
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
    ) -> None:
        """Base regression dataset.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
        """
        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            task="regression",
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )

    @cached_property
    def target_transform(self) -> MinMaxScaler:
        return MinMaxScaler(
            target=self.target_column,
            minimum=self.df_train[self.target_column].min(),
            maximum=self.df_train[self.target_column].max(),
        )


class BaseMTRegressionDataset(AbstractDataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: List[str],
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        batch_size: int,
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
    ) -> None:
        """Base multi-target regression dataset.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
        """
        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            task="regression",
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )

    @cached_property
    def target_transform(self) -> MultiTargetMinMaxScaler:
        return MultiTargetMinMaxScaler(
            targets=self.target_column,
            minimum=self.df_train[self.target_column].min().to_list(),
            maximum=self.df_train[self.target_column].max().to_list(),
        )

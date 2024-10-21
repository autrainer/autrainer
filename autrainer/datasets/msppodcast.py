import os
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig
import pandas as pd
import torch

from autrainer.transforms import SmartCompose

from .abstract_dataset import AbstractDataset
from .aibo import Standardizer
from .utils import (
    AbstractTargetTransform,
    LabelEncoder,
    MinMaxScaler,
)

class MSPPodcast(AbstractDataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[str],
        tracking_metric: str,
        index_column: str,
        target_column: str,
        file_type:str,
        file_handler: Union[str, DictConfig, Dict],
        batch_size: int,
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        standardize: bool = False,
        categories: List[str] = None,
    ) -> None:
        """MSP-Podcast dataset.

        .. warning::
            There are multiple versions available for this dataset.
            We recommend always using the latest one 
            (v1.11 at the time of writing)
            but our code is set up to work with all versions
            (at least up to v1.11).
        
        .. note::
            Note that after v1.7, the dataset features two test sets.
            We only use ``Test1``, as ``Test2`` was found to be biased
            with respect to gender. 
            See https://doi.org/10.21437/Interspeech.2019-1708.

        .. note::
            Unlike other datasets which only support classification
            or regression, MSP-Podcast supports both. This is determined
            by picking the appropriate target column.
            ``EmoClass`` corresponds to categorical emotion classification,
            whereas ``EmoAct``, ``EmoVal``, and ``EmoDom`` to dimensional
            emotion regression for activation (arousal), valence, and dominance,
            respectively.            

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
            standardize: Whether to standardize the data. Defaults to False.
            categories: used to filter out specific emotional categories. 
                Useful for training on subset of data/classes, such as the classic
                ["A", "H", "N", "S"] 4-class problem found in literature. 
                Defaults to None.
        """
        # TODO: expand for multitask regression
        assert target_column in ("EmoClass", "EmoAct", "EmoVal", "EmoDom"), (
            f"{target_column} not included in target list, please choose one of: "
            "[EmoClass, EmoAct, EmoVal, EmoDom]"
        )
        task = "classification" if target_column == "EmoClass" else "regression"
        self.categories = categories
        super().__init__(
            task=task,
            path=path,
            features_subdir=features_subdir,
            seed=seed,
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
        self.standardize = standardize
        if self.standardize:
            train_data = torch.cat([x for x, *_ in self.train_dataset])
            standardizer = Standardizer(
                mean=train_data.mean(0).tolist(),
                std=train_data.std(0).tolist(),
            )

            self.train_transform += standardizer
            self.dev_transform += standardizer
            self.test_transform += standardizer

            self.train_dataset = self._init_dataset(
                self.df_train, self.train_transform
            )

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """
        Download the MSP-Podcast dataset.

        As this dataset is not publicly-available, please download it manually
        by contacting Prof. Carlos Busso: 
        https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html

        This function will not do anything.

        For more information on the data, see:
        https://doi.org/10.1109/TAFFC.2017.2736999
        """
        return None
    
    def load_dataframes(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the dataframes.

        Overrides base class so there is no need
        to create separate train/dev/test files.

        Returns:
            Dataframes for training, development, and testing.
        """
        df = pd.read_csv(os.path.join(self.path, "Labels", "labels_consensus.csv"))
        if self.categories is not None:
            df = df.loc[df["EmoClass"].isin(self.categories)]

        df_train = df.loc[df["Split_Set"] == "Train"]
        df_dev = df.loc[df["Split_Set"] == "Development"]
        df_test = df.loc[df["Split_Set"] == "Test1"]
        return df_train, df_dev, df_test
    
    @cached_property
    def target_transform(self) -> AbstractTargetTransform:
        """Get the target transform.

        Determined automatically based on the type of task.

        Returns:
            Target transform.
        """
        if self.task == "classification":
            return LabelEncoder(
                self.df_train[self.target_column].unique().tolist()
            )
        elif self.task == "regression":
            return MinMaxScaler(
                minimum=self.df_train[self.target_column].min(),
                maximum=self.df_train[self.target_column].max(),
            )
        else:
            raise NotImplementedError(f"{self.task} not supported for MSPPodcast")

    @cached_property
    def output_dim(self) -> int:
        """Get the output dimension of the dataset.

        Determined automatically based on the type of task.

        Returns:
            Number of classes.
        """
        if self.task == "classification":
            return len(self.df_train[self.target_column].unique())
        elif self.task == "regression":
            return 1
        else:
            raise NotImplementedError(f"{self.task} not supported for MSPPodcast")

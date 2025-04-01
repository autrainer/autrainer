from functools import cached_property
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
import pandas as pd

import autrainer
from autrainer.datasets import AbstractDataset
from autrainer.datasets.utils import SSLDatasetWrapper
from autrainer.metrics import AbstractMetric
from autrainer.transforms import SmartCompose


class SSLWrapper(AbstractDataset):
    def __init__(
        self,
        cfg: List[Union[str, DictConfig, Dict]],
        target_transform: Optional[SmartCompose] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
    ):
        self.data = autrainer.instantiate_shorthand(
            config=cfg,
            instance_of=AbstractDataset,
            seed=0,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
        )
        self.target_transform = target_transform
        self.train_transform = self.data.train_transform
        self.dev_transform = self.data.dev_transform
        self.test_transform = self.data.test_transform
        self.features_path = self.data.features_path
        self.features_subdir = self.data.features_subdir
        self.index_column = self.data.index_column
        self.file_type = self.data.file_type
        self.file_handler = self.data.file_handler

    @cached_property
    def output_dim(self) -> int:
        """Mocks the output dimension of the dataset.

        The output dimension
        is determined by the input dimension.
        We set it to `None`
        as SSL-compatible models
        should ignore it.

        Returns:
            Output dimension.
        """
        return None

    def _init_dataset(
        self,
        df: pd.DataFrame,
        transform: SmartCompose,
    ) -> SSLDatasetWrapper:
        """Initialize a wrapper around torch.utils.data.Dataset.

        Args:
            df: Dataframe to use.
            transform: Transform to apply to the features.

        Returns:
            Initialized dataset.
        """
        return SSLDatasetWrapper(
            path=self.features_path,
            features_subdir=self.features_subdir,
            index_column=self.index_column,
            file_type=self.file_type,
            file_handler=self.file_handler,
            df=df,
            transform=transform,
            target_transform=self.target_transform,
        )

    @cached_property
    def target_transform(self) -> SmartCompose:
        return self.target_transform

    @property
    def audio_subdir(self) -> str:
        """Subfolder containing audio data.

        Defaults to `default` for our standard format.
        Should be overridden for datasets
        that do not conform to it.
        """
        return self.data.audio_subdir

    @staticmethod
    def _init_metric(metric: Union[str, DictConfig, Dict]) -> AbstractMetric:
        return autrainer.instantiate_shorthand(
            config=metric,
            instance_of=AbstractMetric,
        )

    @cached_property
    def df_train(self) -> pd.DataFrame:
        """Dataframe for the training set, loaded from `train.csv` by default.

        Returns:
            Training dataframe.
        """
        return self.data.df_train

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        """Dataframe for the development set, loaded from `dev.csv` by default.

        Returns:
            Development dataframe.
        """
        return self.data.df_dev

    @cached_property
    def df_test(self) -> pd.DataFrame:
        """Dataframe for the test set, loaded from `test.csv` by default.

        Returns:
            Test dataframe.
        """
        return self.data.df_test

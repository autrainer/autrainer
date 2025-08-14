from functools import cached_property
import os
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
import pandas as pd

from autrainer.datasets.abstract_dataset import AbstractDataset
from autrainer.datasets.utils import (
    AbstractTargetTransform,
    LabelEncoder,
    MinMaxScaler,
    MultiTargetMinMaxScaler,
)
from autrainer.transforms import SmartCompose


class MSPPodcast(AbstractDataset):
    def __init__(
        self,
        path: str,
        seed: int,
        metrics: List[str],
        tracking_metric: str,
        target_column: str,
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        index_column: str = "FileName",
        features_subdir: Optional[str] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
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
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            index_column: Index column of the dataframe.
                Defaults to `FileName`, as in the original data.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
            categories: used to filter out specific emotional categories.
                Useful for training on subset of data/classes, such as the classic
                ["A", "H", "N", "S"] 4-class problem found in literature.
                Defaults to None.
        """
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
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )

    @property
    def audio_subdir(self) -> str:
        """Subdirectory containing audio data.

        Defaults to `Audios` for MSP-Podcast.
        """
        return "Audios"

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
        return

    @cached_property
    def _df(
        self,
    ) -> pd.DataFrame:
        """Load the dataframes.

        Overrides base class so there is no need
        to create separate train/dev/test files.

        Returns:
            Dataframes for training, development, and testing.
        """
        df = pd.read_csv(os.path.join(self.path, "Labels", "labels_consensus.csv"))
        if self.categories is not None:
            df = df.loc[df["EmoClass"].isin(self.categories)]
        return df.reset_index(drop=True)

    @cached_property
    def df_train(self) -> pd.DataFrame:
        return self._df.loc[self._df["Split_Set"] == "Train"].reset_index(drop=True)

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        return self._df.loc[self._df["Split_Set"] == "Development"].reset_index(
            drop=True
        )

    @cached_property
    def df_test(self) -> pd.DataFrame:
        return self._df.loc[self._df["Split_Set"] == "Test1"].reset_index(drop=True)

    @cached_property
    def target_transform(self) -> AbstractTargetTransform:
        """Get the target transform.

        Determined automatically based on the type of task.

        Returns:
            Target transform.
        """
        if self.task == "classification":
            return LabelEncoder(self.df_train[self.target_column].unique().tolist())
        if self.task == "regression":
            if isinstance(self.target_column, list):
                return MultiTargetMinMaxScaler(
                    target=self.target_column,
                    minimum=self.df_train[self.target_column].min().to_list(),
                    maximum=self.df_train[self.target_column].max().to_list(),
                )
            return MinMaxScaler(
                target=self.target_column,
                minimum=self.df_train[self.target_column].min(),
                maximum=self.df_train[self.target_column].max(),
            )
        raise NotImplementedError(f"{self.task} not supported for MSPPodcast")

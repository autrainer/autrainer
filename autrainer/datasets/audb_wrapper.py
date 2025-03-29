from functools import cached_property
from typing import Dict, List, Optional, Union

import audb
import audformat
from omegaconf import DictConfig
import pandas as pd

from autrainer.datasets.abstract_dataset import AbstractDataset
from autrainer.datasets.utils import (
    AbstractTargetTransform,
    LabelEncoder,
    MinMaxScaler,
    MultiLabelEncoder,
    MultiTargetMinMaxScaler,
)
from autrainer.transforms import SmartCompose


class AudbDataset(AbstractDataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        train_table: str,
        dev_table: str,
        test_table: str,
        name: str,
        audb_params: Dict,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        target_column: str,
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        features_path: Optional[str] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """Audb dataset.

        Database available through `audb`
        (https://audeering.github.io/audb/index.html).

        Currently,
        only supporting the datasets available
        in audEERING's artifactory:
        https://audeering.github.io/datasets/datasets.html

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
                If `None`, defaults to audio subdirectory,
                which is `default` for the standard format,
                but can be overridden in the dataset specification.
            train_table: Table to use for training. Should contain `target_column`.
            dev_table: Table to use for validation. Should contain `target_column`.
            test_table: Table to use for testing. Should contain `target_column`.
            name: Database name. Choose one from
                https://audeering.github.io/datasets/index.html.
            audb_params: Additional `audb.load`.
                See: https://audeering.github.io/audb/api/audb.load.html
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            features_path: Root path to features. Useful
                when features need to be extracted and stored
                in a different folder than the root of the dataset.
                If `None`, will be set to `path`. Defaults to `None`.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
            threshold: Optional threshold for multi-label classification.
                Defaults to None.
        """
        self.train_table = train_table
        self.dev_table = dev_table
        self.test_table = test_table
        audb_params["name"] = name
        audb_params["cache_root"] = path
        self.db = audb.load(**audb_params)
        self.threshold = threshold
        self.target_column = target_column
        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column="file",
            task=self._task,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            features_path=features_path,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )

    @property
    def audio_subdir(self) -> str:
        """Subfolder containing audio data."""
        return ""

    @cached_property
    def df_train(self) -> pd.DataFrame:
        return self.db.tables[self.train_table].df.reset_index()

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        return self.db.tables[self.dev_table].df.reset_index()

    @cached_property
    def df_test(self) -> pd.DataFrame:
        return self.db.tables[self.test_table].df.reset_index()

    @cached_property
    def _task(self) -> str:
        if isinstance(self.target_column, list):
            # either MTC or MLC
            # check type of first target
            scheme = self.db.tables[self.train_table][
                self.target_column[0]
            ].scheme_id
            format = self.db.schemes[scheme].dtype
            if format == audformat.define.DataType.FLOAT:
                return "mt-classification"
            elif format in (
                audformat.define.DataType.BOOL,
                audformat.define.DataType.INTEGER,
                audformat.define.DataType.STRING,
            ):
                return "ml-classification"
            else:
                raise NotImplementedError(f"{format} not supported.")
        else:
            scheme = self.db.tables[self.train_table][
                self.target_column
            ].scheme_id
            format = self.db.schemes[scheme].dtype
            if format == audformat.define.DataType.FLOAT:
                return "regression"
            elif format in (
                audformat.define.DataType.BOOL,
                audformat.define.DataType.INTEGER,
                audformat.define.DataType.STRING,
            ):
                return "classification"
            else:
                raise NotImplementedError(f"{format} not supported.")

    @property
    def target_transform(self) -> AbstractTargetTransform:
        """Get the transform to apply to the target.

        Returns:
            Target transform.
        """
        if self.task == "classification":
            return LabelEncoder(
                self.db.tables[self.train_table]
                .df[self.target_column]
                .unique()
                .tolist()
            )
        elif self.task == "regression":
            return MinMaxScaler(
                target=self.target_column,
                minimum=self.db.tables[self.train_table]
                .df[self.target_column]
                .min(),
                maximum=self.db.tables[self.train_table]
                .df[self.target_column]
                .max(),
            )
        elif self.task == "ml-classification":
            assert self.threshold is not None, "Threshold should not be None."
            return MultiLabelEncoder(
                threshold=self.threshold, labels=self.target_column
            )
        elif self.task == "mt-regression":
            return MultiTargetMinMaxScaler(
                target=self.target_column,
                minimum=self.db.tables[self.train_table]
                .df[self.target_column]
                .min(),
                maximum=self.db.tables[self.train_table]
                .df[self.target_column]
                .max(),
            )

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download database.

        `audb` implements its own download management.
        This is tethered to the dataset version
        and some configuration parameters
        like the sampling rate or the number of channels.
        This is not compatible
        with the automatic download workflow
        of `autrainer`.
        The database will be downloaded
        during instantiation
        when calling `autrainer preprocess`
        or `autrainer train`.

        Args:
            path: Path to the directory to download the dataset to.
        """
        pass


if __name__ == "__main__":
    data = AudbDataset(
        path="data",
        features_subdir=None,
        train_table="emotion.categories.train.gold_standard",
        dev_table="emotion.categories.train.gold_standard",
        test_table="emotion.categories.test.gold_standard",
        name="emodb",
        file_handler="autrainer.datasets.utils.AudioFileHandler",
        audb_params={
            "version": "1.4.1",
            "sampling_rate": 16000,
            "mixdown": True,
            "format": "wav",
        },
        seed=0,
        target_column="emotion",
        metrics=["autrainer.metrics.Accuracy"],
        tracking_metric="autrainer.metrics.Accuracy",
        file_type="wav",
    )
    loader = data.create_train_loader(1)
    batch = next(iter(loader))
    print(batch)

    data = AudbDataset(
        path="data",
        features_subdir=None,
        train_table="emotion.categories.train.gold_standard",
        dev_table="emotion.categories.train.gold_standard",
        test_table="emotion.categories.test.gold_standard",
        name="emodb",
        file_handler="autrainer.datasets.utils.AudioFileHandler",
        audb_params={
            "version": "1.4.1",
            "sampling_rate": 16000,
            "mixdown": True,
            "format": "wav",
        },
        seed=0,
        target_column="emotion.confidence",
        metrics=["autrainer.metrics.CCC"],
        tracking_metric="autrainer.metrics.CCC",
        file_type="wav",
    )
    loader = data.create_train_loader(1)
    batch = next(iter(loader))
    print(batch)

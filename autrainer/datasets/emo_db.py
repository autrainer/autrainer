from functools import cached_property
import os
import shutil
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
import pandas as pd

from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseClassificationDataset
from .utils import ZipDownloadManager


FILES = {"download.zip": "http://emodb.bilderbar.info/download/download.zip"}

EMOTIONS = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}


class EmoDB(BaseClassificationDataset):
    def __init__(
        self,
        path: str,
        features_subdir: Optional[str],
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: str,
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        features_path: Optional[str] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        train_speakers: Optional[List[int]] = None,
        dev_speakers: Optional[List[int]] = None,
        test_speakers: Optional[List[int]] = None,
    ) -> None:
        """EmoDB dataset for the task of Speech Emotion Recognition.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
                If `None`, defaults to audio subdirectory,
                which is `default` for the standard format,
                but can be overridden in the dataset specification.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            features_path: Root path to features. Useful
                when features need to be extracted and stored
                in a different directory than the root of the dataset.
                If `None`, will be set to `path`. Defaults to `None`.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
            train_speakers: List of speakers IDs (int) to use for training.
                If None, 3, 8, 9, 10, 11, 12 are used. Defaults to None.
            dev_speakers: List of speakers IDs (int) to use for validation.
                If None, 13, 14 are used. Defaults to None.
            test_speakers: List of speakers IDs (int) to use for testing.
                If None, 15, 16 are used. Defaults to None.
        """
        self.train_speakers = train_speakers or [3, 8, 9, 10, 11, 12]
        self.dev_speakers = dev_speakers or [13, 14]
        self.test_speakers = test_speakers or [15, 16]
        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            features_path=features_path,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )

    @cached_property
    def meta(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.path, "metadata.csv"))

    @cached_property
    def df_train(self) -> pd.DataFrame:
        return self.meta[self.meta["speaker"].isin(self.train_speakers)]

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        return self.meta[self.meta["speaker"].isin(self.dev_speakers)]

    @cached_property
    def df_test(self) -> pd.DataFrame:
        return self.meta[self.meta["speaker"].isin(self.test_speakers)]

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the EmoDB dataset.

        For more information on the dataset, see:
        http://emodb.bilderbar.info/docu/

        Args:
            path: Path to the directory to download the dataset to.
        """

        out_path = os.path.join(path, "default")
        if os.path.isdir(out_path):
            return

        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(check_exist=["download.zip"])
        dl_manager.extract(check_exist=["wav"])
        shutil.move(os.path.join(path, "wav"), out_path)
        files = os.listdir(out_path)
        files = sorted(files)

        df = pd.DataFrame(files, columns=["filename"])
        df["emotion"] = df["filename"].str[5].map(EMOTIONS)
        df["speaker"] = df["filename"].str[0:2]
        df["text"] = df["filename"].str[2:5]
        df["version"] = df["filename"].str[6]

        df.to_csv(os.path.join(path, "metadata.csv"), index=False)

        # remove unnecessary files
        shutil.rmtree(os.path.join(path, "lablaut"))
        shutil.rmtree(os.path.join(path, "labsilb"))
        shutil.rmtree(os.path.join(path, "silb"))
        os.remove(os.path.join(path, "erkennung.txt"))
        os.remove(os.path.join(path, "erklaerung.txt"))

from functools import cached_property
import os
import shutil
from typing import Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig
import pandas as pd

from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseSEDDataset
from .utils import SEDEncoder, ZipDownloadManager


FILES = {
    "Synthetic_dataset.zip": "https://zenodo.org/records/2583796/files/Synthetic_dataset.zip?download=1",
    "DESED_public_eval.tar.gz": "https://zenodo.org/records/3588172/files/DESEDpublic_eval.tar.gz?download=1",
}

EVENTS = [
    "Speech",
    "Dog",
    "Cat",
    "Alarm_bell_ringing",
    "Dishes",
    "Frying",
    "Blender",
    "Running_water",
    "Vacuum_cleaner",
    "Electric_shaver_toothbrush",
]


class DCASE2019Task4(BaseSEDDataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        file_type: str,
        index_column: str,
        target_column: str,
        file_handler: Union[str, DictConfig, Dict],
        batch_size: int = 1,
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        frame_rate: float = 0.08,
        duration: float = 10.0,
        threshold: float = 0.3,
        min_event_length: float = 0.3,
        pause_length: float = 0.5,
    ) -> None:
        """DCASE 2019 Task 4 dataset.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
            frame_rate: Frame rate in seconds. Defaults to 0.08.
            duration: Duration of each audio segment in seconds. Defaults to 10.0.
            threshold: Threshold for event detection. Defaults to 0.3.
            min_event_length: Minimum length of an event in seconds. Defaults to 0.3.
            pause_length: Minimum pause length between events in seconds. Defaults to 0.5.
        """
        self.onset_column = "onset"
        self.offset_column = "offset"
        self.frame_rate = frame_rate
        self.num_frames = int(np.ceil(duration / frame_rate))
        self.durations = {
            "frame_rate": frame_rate,
            "duration": duration,
            "threshold": threshold,
            "min_event_length": min_event_length,
            "pause_length": pause_length,
        }
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
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
        )

    @cached_property
    def target_transform(self) -> SEDEncoder:
        return SEDEncoder(EVENTS, **self.durations)

    def _framewise_representation(self, df):
        """Transform data to framewise representations.

        Each row in the dataframe
        corresponds to a file
        with optional start/end times
        and a 2D matrix
        with 0s and 1s
        marking event presence
        in each frame
        for all events in the dataset.
        """

        def collect_frames(df):
            results = np.zeros(
                (self.num_frames, len(self.target_transform.labels))
            )
            for _, row in df.iterrows():
                start = int(row[self.onset_column] / self.frame_rate)
                end = int(row[self.offset_column] / self.frame_rate)
                results[
                    start:end,
                    self.target_transform.encode(row[self.target_column]),
                ] = 1
            return results

        df = (
            df.groupby(self.index_column)
            .apply(collect_frames)
            .to_frame()
            .rename(columns={0: self.target_column})
            .reset_index()
        )
        return df

    @property
    def audio_subdir(self) -> str:
        """Subfolder containing audio data."""
        return ""

    @cached_property
    def df_train(self):
        df = pd.read_csv(
            os.path.join(self.path, "synthetic_dataset.csv"), sep="\t"
        )
        df["filename"] = df["filename"].apply(
            lambda x: os.path.join("audio", "train", "synthetic", x)
        )
        return self._framewise_representation(df)

    @cached_property
    def df_dev(self):
        return self.df_test

    @cached_property
    def df_test(self):
        df = pd.read_csv(
            os.path.join(
                self.path, "dataset", "metadata", "eval", "public.tsv"
            ),
            sep="\t",
        )
        df["filename"] = df["filename"].apply(
            lambda x: os.path.join("dataset", "audio", "eval", "public", x)
        )
        return self._framewise_representation(df)

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the DCASE 2019 Task 4 dataset.

        For more information on the dataset and dataset split, see:
        https://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments

        Args:
            path: Path to download the dataset to
            create_windows: If True, creates fixed windows during download
        """

        # download and extract files
        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(
            check_exist=["Synthetic_dataset", "DESED_public_eval"]
        )
        dl_manager.extract(
            check_exist=["Synthetic_dataset", "DESED_public_eval"]
        )

        def remove_if_exists(path: str) -> None:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        for archive in ["DESED_public_eval.tar.gz", "Synthetic_dataset.zip"]:
            remove_if_exists(os.path.join(path, archive))

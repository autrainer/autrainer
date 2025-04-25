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
    # Development dataset - Synthetic data ~1 GB
    "Synthetic_dataset.zip": "https://zenodo.org/records/2583796/files/Synthetic_dataset.zip?download=1",
    # TODO: Development set - Weak data, unlabel_in_domain_dev_path -requires youtube-dl package
    # https://github.com/turpaultn/DESED/blob/master/desed/desed/download.py
    # https://github.com/turpaultn/DCASE2019_task4/blob/public/baseline/download_data.py
    # Evaluation dataset - Public evaluation set ~1 GB
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

DURATIONS = {
    "min_dur_event": 0.25,
    "min_dur_inter": 0.15,
    "length_sec": 10,
}


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
        """
        self.onset_column = "onset"
        self.offset_column = "offset"
        # FIXME: make more modular
        self.duration = 10  # duration fixed @ 10s
        self.frame_rate = frame_rate
        assert self.frame_rate is not None
        assert self.frame_rate > 0
        # FIXME: Maybe np.ceil?
        self.num_frames = int(self.duration / self.frame_rate)
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
        return SEDEncoder(EVENTS)

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

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
    "TUT-sound-events-2017-development.audio.1.zip": "https://zenodo.org/records/814831/files/TUT-sound-events-2017-development.audio.1.zip?download=1",
    "TUT-sound-events-2017-development.audio.2.zip": "https://zenodo.org/records/814831/files/TUT-sound-events-2017-development.audio.2.zip?download=1",
    "TUT-sound-events-2017-development.meta.zip": "https://zenodo.org/records/814831/files/TUT-sound-events-2017-development.meta.zip?download=1",
    "TUT-sound-events-2017-evaluation.audio.zip": "https://zenodo.org/records/1040179/files/TUT-sound-events-2017-evaluation.audio.zip?download=1",
    "TUT-sound-events-2017-evaluation.meta.zip": "https://zenodo.org/records/1040179/files/TUT-sound-events-2017-evaluation.meta.zip?download=1",
}

EVENTS = [
    "brakes squeaking",
    "car",
    "children",
    "large vehicle",
    "people speaking",
    "people walking",
]


class DCASE2017Task3(BaseSEDDataset):
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
        fold: int = 1,
        frame_rate: float = 0.08,
        duration: float = 10.0,
        threshold: float = 0.3,
        min_event_length: float = 0.3,
        pause_length: float = 0.5,
    ) -> None:
        """DCASE 2017 Task 3 dataset.

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
            fold: Which fold to use in [1, 2, 3, 4]. Defaults to 1.
            frame_rate: Frame rate in seconds. Defaults to 0.08.
            duration: Duration of each audio segment in seconds. Defaults to 10.0.
            threshold: Threshold for event detection. Defaults to 0.3.
            min_event_length: Minimum length of an event in seconds. Defaults to 0.3.
            pause_length: Minimum pause length between events in seconds. Defaults to 0.5.
        """
        self._assert_choice(fold, [1, 2, 3, 4], "fold")
        self.fold = fold
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
        df = pd.read_csv(os.path.join(self.path, f"fold{self.fold}_train.csv"))
        return self._framewise_representation(df)

    @cached_property
    def df_dev(self):
        df = pd.read_csv(
            os.path.join(self.path, f"fold{self.fold}_evaluate.csv")
        )
        return self._framewise_representation(df)

    @cached_property
    def df_test(self):
        df = pd.read_csv(os.path.join(self.path, "test.csv"))
        return self._framewise_representation(df)

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the DCASE 2017 Task 3 dataset.

        For more information on the dataset and dataset split, see:
        https://dcase.community/challenge2017/task-sound-event-detection-in-real-life-audio

        Args:
            path: Path to download the dataset to.
        """
        out_path = os.path.join(path, "default")
        if os.path.isdir(out_path):
            return
        os.makedirs(out_path, exist_ok=True)

        # download and extract files
        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(
            check_exist=[
                "TUT-sound-events-2017-development",
                "TUT-sound-events-2017-evaluation",
            ]
        )
        dl_manager.extract(
            check_exist=[
                "TUT-sound-events-2017-development",
                "TUT-sound-events-2017-evaluation",
            ]
        )

        # move audio files to the same directory
        dev_path = os.path.join(path, "TUT-sound-events-2017-development")
        eval_path = os.path.join(path, "TUT-sound-events-2017-evaluation")
        for file in os.listdir(os.path.join(dev_path, "audio")):
            shutil.move(
                os.path.join(os.path.join(dev_path, "audio"), file), out_path
            )
        for file in os.listdir(os.path.join(eval_path, "audio", "street")):
            shutil.move(
                os.path.join(os.path.join(eval_path, "audio", "street"), file),
                out_path,
            )

        # load dev folds
        for fold in range(1, 5):
            df_train = pd.read_csv(
                os.path.join(
                    dev_path,
                    "evaluation_setup",
                    f"street_fold{fold}_train.txt",
                ),
                sep="\t",
                names=["filename", "onset", "offset", "event_label"],
                usecols=[0, 2, 3, 4],
            )
            df_train["filename"] = df_train["filename"].apply(
                lambda x: x.split("/")[-1]
            )
            df_train.to_csv(
                os.path.join(path, f"fold{fold}_train.csv"),
                index=False,
            )
            df_evaluate = pd.read_csv(
                os.path.join(
                    dev_path,
                    "evaluation_setup",
                    f"street_fold{fold}_evaluate.txt",
                ),
                sep="\t",
                names=["filename", "onset", "offset", "event_label"],
                usecols=[0, 2, 3, 4],
            )
            df_evaluate["filename"] = df_evaluate["filename"].apply(
                lambda x: x.split("/")[-1]
            )
            df_evaluate.to_csv(
                os.path.join(path, f"fold{fold}_evaluate.csv"),
                index=False,
            )

        # eval set
        df_eval = pd.read_csv(
            os.path.join(eval_path, "evaluation_setup", "street_evaluate.txt"),
            sep="\t",
            names=["filename", "onset", "offset", "event_label"],
            usecols=[0, 2, 3, 4],
        )
        df_eval["filename"] = df_eval["filename"].apply(
            lambda x: x.split("/")[-1]
        )
        df_eval.to_csv(os.path.join(path, "test.csv"), index=False)

        # remove unnecessary files
        shutil.rmtree(dev_path)
        shutil.rmtree(eval_path)
        for zip_file in FILES.keys():
            zip_path = os.path.join(path, zip_file)
            if os.path.exists(zip_path):
                os.remove(zip_path)
        street_path = os.path.join(out_path, "street")
        if os.path.exists(street_path):
            for file in os.listdir(street_path):
                shutil.move(os.path.join(street_path, file), out_path)
            shutil.rmtree(street_path)

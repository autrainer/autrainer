from functools import cached_property
import os
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
import pandas as pd

from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseClassificationDataset


class AIBO(BaseClassificationDataset):
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
        aibo_task: str = "2cl",
    ) -> None:
        """FAU AIBO dataset.

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
            aibo_task: Task to load in ["2cl", "5cl"]. Defaults to "2cl".
        """
        self._assert_choice(aibo_task, ["2cl", "5cl"], "aibo_task")
        self.aibo_task = aibo_task
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

    @property
    def audio_subdir(self) -> str:
        """Subdirectory containing audio data."""
        return "wav"

    @cached_property
    def _load_df(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.path, f"chunk_labels_{self.aibo_task}_corpus.txt"),
            header=None,
            sep=" ",
        )
        df = df.rename(columns={0: "id", 1: "class", 2: "conf"})
        df["file"] = df["id"].apply(lambda x: x + ".wav")
        df["school"] = df["id"].apply(lambda x: x.split("_")[0])
        df["speaker"] = df["id"].apply(lambda x: x.split("_")[1])
        return df

    @cached_property
    def df_train(self) -> pd.DataFrame:
        df = self._load_df
        df_train_dev = df.loc[df["school"] == "Ohm"]
        speakers = sorted(df_train_dev["speaker"].unique())
        return df_train_dev.loc[df_train_dev["speaker"].isin(speakers[:-2])]

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        df = self._load_df
        df_train_dev = df.loc[df["school"] == "Ohm"]
        speakers = sorted(df_train_dev["speaker"].unique())
        return df_train_dev.loc[df_train_dev["speaker"].isin(speakers[-2:])]

    @cached_property
    def df_test(self) -> pd.DataFrame:
        df = self._load_df
        return df.loc[df["school"] == "Mont"]

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """
        Download the FAU AIBO dataset.

        As the AIBO dataset is private, this method does not download the
        dataset but rather prepares the file structure expected by the
        preprocessing routines.

        In the specified path, the following directories and files are expected:

        - `default/`: Directory containing .wav files.
        - `chunk_labels_2cl_corpus.txt`: File containing the file names and
          corresponding labels for the 2-class classification task.
        - `chunk_labels_5cl_corpus.txt`: File containing the file names and
          corresponding labels for the 5-class classification task.

        For more information on the dataset and dataset split, see:
        https://doi.org/10.1109/ICME51207.2021.9428217

        Args:
            path: Path to the directory to download the dataset to.
        """
        if not os.path.isfile(
            os.path.join(path, "chunk_labels_2cl_corpus.txt")
            or os.path.isfile(os.path.join(path, "chunk_labels_5cl_corpus.txt"))
        ):
            raise ValueError(
                f"File 'chunk_labels_2cl_corpus.txt' or "
                f"'chunk_labels_5cl_corpus.txt' does not exist in '{path}'."
            )

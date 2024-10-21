import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig
import pandas as pd

from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseClassificationDataset
from .utils import ZipDownloadManager


FILES = {
    "ff1010bird_wav.zip": "https://archive.org/download/ff1010bird/ff1010bird_wav.zip",
    "ff1010bird_metadata_2018.csv": "https://ndownloader.figshare.com/files/10853303",
    "BirdVox-DCASE-20k.zip": "https://zenodo.org/record/1208080/files/BirdVox-DCASE-20k.zip",
    "BirdVoxDCASE20k_csvpublic.csv": "https://ndownloader.figshare.com/files/10853300",
    "warblrb10k_public_wav.zip": "https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip",
    "warblrb10k_public_metadata_2018.csv": "https://ndownloader.figshare.com/files/10853306",
}


class DCASE2018Task3(BaseClassificationDataset):
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
        dev_split: float = 0.0,
        dev_split_seed: Optional[int] = None,
    ) -> None:
        """DCASE 2018 Task 3 dataset.

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
            dev_split: Fraction of the training set to use as the development
                set. Defaults to 0.0.
            dev_split_seed: Seed for the development split. If None, seed is
                used. Defaults to None.
        """
        self._assert_dev_split(dev_split)
        self.dev_split = dev_split
        self.dev_split_seed = dev_split_seed or seed
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
            stratify=stratify,
        )

    def load_dataframes(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_train = pd.read_csv(os.path.join(self.path, "train.csv"))
        df_dev = pd.read_csv(os.path.join(self.path, "test.csv"))
        df_test = pd.read_csv(os.path.join(self.path, "test.csv"))

        if self.dev_split > 0:
            df_train, df_dev = self._split_train_dataset(df_train)

        return df_train, df_dev, df_test

    def _split_train_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.dev_split_seed)
        indices = rng.permutation(df.index)
        train_indices = indices[: int(len(indices) * (1 - self.dev_split))]
        return (
            df[df.index.isin(train_indices)].copy(),
            df[~df.index.isin(train_indices)].copy(),
        )

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the DCASE 2018 Task 3 dataset.

        For the train dataset, the following subsets are used:

        - Field recordings, worldwide ("freefield1010")
        - Remote monitoring flight calls, USA ("BirdVox-DCASE-20k")

        For the test dataset, the following subset is used:

        - Crowdsourced dataset, UK ("warblrb10k")

        Both the training and test datasets are taken from the development set,
        as no labels are provided for the evaluation set.
        For more information on the dataset, see:
        https://dcase.community/challenge2018/task-bird-audio-detection

        Args:
            path: Path to the directory to download the dataset to.
        """

        out_path = os.path.join(path, "default")
        if os.path.isdir(out_path):
            return

        # download and extract files
        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(check_exist=["wav", "warblrb10k_public_wav"])
        dl_manager.extract(check_exist=["wav", "warblrb10k_public_wav"])

        # move audio files
        shutil.move(os.path.join(path, "wav"), out_path)

        # process dataframes
        ff1010_path = os.path.join(path, "ff1010bird_metadata_2018.csv")
        birdvox_path = os.path.join(path, "BirdVoxDCASE20k_csvpublic.csv")
        warblrb_path = os.path.join(
            path, "warblrb10k_public_metadata_2018.csv"
        )
        df_ff1010 = pd.read_csv(ff1010_path).rename(
            columns={"itemid": "filename", "datasetid": "dataset"}
        )
        df_birdvox = pd.read_csv(birdvox_path).rename(
            columns={"itemid": "filename", "datasetid": "dataset"}
        )
        df_warblrb10k = pd.read_csv(warblrb_path).rename(
            columns={"itemid": "filename", "datasetid": "dataset"}
        )
        df_train = pd.concat([df_ff1010, df_birdvox])
        df_test = df_warblrb10k
        df_train["filename"] = df_train["filename"].apply(lambda x: f"{x}.wav")
        df_test["filename"] = df_test["filename"].apply(lambda x: f"{x}.wav")
        df_train.to_csv(os.path.join(path, "train.csv"), index=False)
        df_test.to_csv(os.path.join(path, "test.csv"), index=False)

        # remove unnecessary files
        os.remove(ff1010_path)
        os.remove(birdvox_path)
        os.remove(warblrb_path)

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
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.1.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.1.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.2.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.2.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.3.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.3.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.4.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.4.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.5.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.5.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.6.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.6.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.7.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.7.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.8.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.8.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.9.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.9.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.10.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.10.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.11.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.11.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.12.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.12.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.13.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.13.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.14.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.14.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.15.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.15.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.audio.16.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.16.zip?download=1",
    "TAU-urban-acoustic-scenes-2020-mobile-development.meta.zip": "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.meta.zip?download=1",
}

SCENE_CATEGORIES = {
    "indoor": ["airport", "shopping_mall", "metro_station"],
    "outdoor": [
        "park",
        "public_square",
        "street_pedestrian",
        "street_traffic",
    ],
    "transportation": ["bus", "metro", "tram"],
}


class DCASE2020Task1A(BaseClassificationDataset):
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
        scene_category: Optional[str] = None,
        exclude_cities: Optional[List[str]] = None,
    ) -> None:
        """TAU Urban Acoustic Scenes 2020 Mobile Task 1 Subtask A
        (DCASE2020Task1A) dataset.

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
            scene_category: Scene category in ["indoor", "outdoor",
                "transportation"]. Defaults to None.
            exclude_cities: List of cities to exclude from the dataset.
                Defaults to None.
        """
        self._assert_dev_split(dev_split)
        self.dev_split = dev_split
        self.dev_split_seed = dev_split_seed or seed
        self._assert_scene_category(scene_category)
        self.scene_category = scene_category
        self.exclude_cities = exclude_cities
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

    @staticmethod
    def _assert_scene_category(scene_category: Optional[str]) -> None:
        if scene_category is None:
            return
        if scene_category not in SCENE_CATEGORIES.keys():
            raise ValueError(
                f"Scene category '{scene_category}' must be one of "
                f"{list(SCENE_CATEGORIES.keys())}."
            )

    def load_dataframes(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_train = pd.read_csv(os.path.join(self.path, "train.csv"))
        df_dev = pd.read_csv(os.path.join(self.path, "test.csv"))
        df_test = pd.read_csv(os.path.join(self.path, "test.csv"))

        if self.dev_split > 0:
            df_train, df_dev = self._split_train_dataset(df_train)

        df_train = self._filter_df_by_category(df_train, self.scene_category)
        df_dev = self._filter_df_by_category(df_dev, self.scene_category)
        df_test = self._filter_df_by_category(df_test, self.scene_category)

        if self.exclude_cities is not None:
            df_train = df_train.loc[
                ~df_train["city"].isin(self.exclude_cities)
            ]

        return df_train, df_dev, df_test

    def _split_train_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the training dataset into training and development sets
        taking into account the location.
        Data is split to include at least the `dev_split` fraction of the
        training data in the development set, e.g. favoring the development
        set and the split is not exactly `dev_split`.


        Args:
            df (pd.DataFrame): Training dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and development
            datasets.
        """
        self.rng = np.random.default_rng(self.dev_split_seed)
        unique_locations = df["location"].unique()
        self.rng.shuffle(unique_locations)
        train_size = len(df)
        train_locations = []
        dev_locations = []
        dev_size = 0

        for loc in unique_locations:
            if dev_size < train_size * self.dev_split:
                dev_locations.append(loc)
                dev_size += len(df[df["location"] == loc])
            else:
                train_locations.append(loc)

        return (
            df[df["location"].isin(train_locations)].copy(),
            df[df["location"].isin(dev_locations)].copy(),
        )

    def _filter_df_by_category(
        self,
        df: pd.DataFrame,
        category: Optional[str],
    ) -> pd.DataFrame:
        if category is None:
            return df
        return df.loc[df[self.target_column].isin(SCENE_CATEGORIES[category])]

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the TAU Urban Acoustic Scenes 2020 Mobile Task 1 Subtask A
        (DCASE2020Task1A) dataset.

        As no labels are provided for the evaluation set, the provided training
        and test split of the development set is created.
        Therefore, this download does not include the evaluation set.

        For more information on the dataset, see:
        https://dcase.community/challenge2020/task-acoustic-scene-classification

        Args:
            path: Path to the directory to download the dataset to.
        """

        def _extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
            df["city"] = df["filename"].apply(lambda x: x.split("-")[1])
            df["location"] = df["filename"].apply(lambda x: x.split("-")[2])
            df["segment"] = df["filename"].apply(lambda x: x.split("-")[3])
            df["device"] = df["filename"].apply(
                lambda x: x.split("-")[4].split(".")[0]
            )
            return df

        out_path = os.path.join(path, "default")
        if os.path.isdir(out_path):
            return
        os.makedirs(out_path, exist_ok=True)

        # download and extract files
        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(
            check_exist=["TAU-urban-acoustic-scenes-2020-mobile-development"]
        )
        dl_manager.extract(
            check_exist=["TAU-urban-acoustic-scenes-2020-mobile-development"]
        )
        dev_path = os.path.join(
            path, "TAU-urban-acoustic-scenes-2020-mobile-development"
        )

        # move audio files
        shutil.move(
            os.path.join(dev_path, "audio"),
            os.path.join(out_path),
        )

        # load, extract, and save dataframes
        df_train = pd.read_csv(
            os.path.join(dev_path, "evaluation_setup", "fold1_train.csv"),
            sep="\t",
        )
        df_test = pd.read_csv(
            os.path.join(dev_path, "evaluation_setup", "fold1_evaluate.csv"),
            sep="\t",
        )
        df_train = _extract_metadata(df_train)
        df_test = _extract_metadata(df_test)
        df_train.to_csv(os.path.join(path, "train.csv"), index=False)
        df_test.to_csv(os.path.join(path, "test.csv"), index=False)

        # remove unnecessary files
        shutil.rmtree(dev_path)

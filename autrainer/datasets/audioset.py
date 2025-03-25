import copy
from functools import cached_property
import json
import os
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
import pandas as pd

from autrainer.datasets.abstract_dataset import BaseMLClassificationDataset
from autrainer.transforms import SmartCompose


def flatten_list(nested_list: list) -> list:
    """Flatten an arbitrarily nested list.

    Implemented without  recursion to avoid stack overflows.
    Returns a new list, the original list is unchanged.

    Args:
        nested_list: nested list

    Returns:
        flattened list

    Examples:
        >>> audeer.flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]])
        [1, 2, 3, 4, 5]
        >>> audeer.flatten_list([[1, 2], 3])
        [1, 2, 3]

    """

    def _flat_generator(nested_list):
        while nested_list:
            sublist = nested_list.pop(0)
            if isinstance(sublist, list):
                nested_list = sublist + nested_list
            else:
                yield sublist

    nested_list = copy.deepcopy(nested_list)
    return list(_flat_generator(nested_list))


class AudioSet(BaseMLClassificationDataset):
    categories = {
        "Human sounds": [
            "Human voice",
            "Whistling",
            "Respiratory sounds",
            "Human locomotion",
            "Digestive",
            "Hands",
            "Heart sounds, heartbeat",
            "Otoacoustic emission",
            "Human group actions",
        ],
        "Source-ambiguous sounds": [
            "Generic impact sounds",
            "Surface contact",
            "Deformable shell",
            "Onomatopoeia",
            "Silence",
            "Other sourceless",
        ],
        "Animal": [
            "Domestic animals, pets",
            "Livestock, farm animals, working animals",
            "Wild animals",
        ],
        "Sounds of things": [
            "Vehicle",
            "Engine",
            "Domestic sounds, home sounds",
            "Bell",
            "Alarm",
            "Mechanisms",
            "Tools",
            "Explosion",
            "Wood",
            "Glass",
            "Liquid",
            "Miscellaneous sources",
            "Specific impact sounds",
        ],
        "Music": [
            "Musical instrument",
            "Music genre",
            "Musical concepts",
            "Music role",
            "Music mood",
        ],
        "Natural sounds": ["Wind", "Thunderstorm", "Water", "Fire"],
        "Channel, environment and background": [
            "Acoustic environment",
            "Noise",
            "Sound reproduction",
        ],
    }

    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        target_column: List[str],
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        batch_size: int,
        inference_batch_size: Optional[int] = None,
        features_path: Optional[str] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        threshold: float = 0.5,
        use_unbalanced: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> None:
        """AudioSet dataset.

        .. warning::

            AudioSet changes constantly
            as videos are removed from
            YouTube over time.
            Results are not reproducible
            across different snapshots
            of the data.

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
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
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
            threshold: Threshold for classification. Defaults to 0.5.
        """
        self.use_unbalanced = use_unbalanced
        self.include = include
        self.exclude = exclude
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
            features_path=features_path,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
            threshold=threshold,
        )

    def _filename_and_ids(self, df):
        r"""Return data frame with filenames and IDs.

        Args:
            df (pandas.DataFrame): data frame as read in from the CSV file

        Results:
            pandas.DataFrame: data frame with columns `filename` and `ids`

        """
        df = df.rename(columns={"positive_labels": "ids"})
        # Translate labels from "label1,label2" to [label1, label2]
        df["ids"] = [label.strip('"').split(",") for label in df["ids"]]
        # Insert filename
        df["filename"] = (
            df["# YTID"]
            + "_"
            + [f"{x:.3f}" for x in df["start_seconds"]]
            + ".wav"
        )
        return df[["filename", "ids"]]

    def _add_parent_ids(self, child_ids):
        r"""Add all parent IDs to the list of given child IDs.

        Args:
            child_ids (list of str): child IDs

        Return:
            list of str: list of child and parent IDs

        """
        ids = child_ids
        for id in child_ids:
            ids += [x["id"] for x in self.ontology if id in x["child_ids"]]
        # Remove duplicates
        return list(set(ids))

    def _convert_ids_to_categories(self, ids):
        r"""Convert list of ids to sorted list of categories.

        Args:
            ids (list of str): list of IDs

        Returns:
            list of str: list of sorted categories

        """
        # Convert IDs to categories
        categories = []
        for id in ids:
            categories += [x["name"] for x in self.ontology if x["id"] == id]
        # Order categories after the first two top ontologies
        order = []
        first_hierarchy = self.categories.keys()
        second_hierarchy = flatten_list(list(self.categories.values()))
        for cat in categories:
            if cat in first_hierarchy:
                order += [0]
            elif cat in second_hierarchy:
                order += [1]
            else:
                order += [2]
        # Sort list `categories` by the list `order`
        categories = [cat for _, cat in sorted(zip(order, categories))]
        return categories

    def _filter_by_categories(
        self,
        df,
        categories,
        exclude_mode=False,
    ):
        r"""Return data frame containing only specified categories.

        Args:
            df (pandas.DataFrame): data frame containing the columns `ids`
            categories (list of str): list of categories to include or exclude
            exclude_mode (bool, optional): if `False` the specified categories
                should be included in the data frame, otherwise excluded.
                Default: `False`

        Returns:
            pandas.DataFrame: data frame containing only the desired categories

        """
        ids = self._ids_for_categories(categories)
        if exclude_mode:
            # Remove rows that have an intersection of actual and desired IDs
            df = df[
                [
                    False if set(row["ids"]) & set(ids) else True
                    for _, row in df.iterrows()
                ]
            ]
        else:
            # Include rows that have an intersection of actual and desired IDs
            df = df[
                [
                    True if set(row["ids"]) & set(ids) else False
                    for _, row in df.iterrows()
                ]
            ]
        df = df.reset_index(drop=True)
        return df

    def _ids_for_categories(self, categories):
        r"""All IDs and child IDs for a given set of categories.

        Args:
            categories (list of str): list of categories

        Returns:
            list: list of IDs

        """
        ids = []
        category_ids = [
            x["id"] for x in self.ontology if x["name"] in categories
        ]
        for category_id in category_ids:
            ids += self._subcategory_ids(category_id)
        # Remove duplicates
        return list(set(ids))

    def _subcategory_ids(self, parent_id):
        r"""Recursively identify all IDs of a given category.

        Args:
            parent_id (unicode str): ID of parent category

        Returns:
            list: list of all children IDs and the parent ID

        """
        id_list = [parent_id]
        child_ids = [
            x["child_ids"] for x in self.ontology if x["id"] == parent_id
        ]
        child_ids = flatten_list(child_ids)
        # Add all subcategories
        for child_id in child_ids:
            id_list += self._subcategory_ids(child_id)
        return id_list

    def map_to_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._filename_and_ids(df)
        if self.include is not None:
            df = self._filter_by_categories(df, self.include)
        df["ids"] = df["ids"].map(self._add_parent_ids)
        if self.exclude is not None:
            df = self._filter_by_categories(
                df, self.exclude, exclude_mode=True
            )
        categories = df["ids"].map(self._convert_ids_to_categories)
        print(categories)
        print(df.head())
        exit()
        pass

    @cached_property
    def ontology(self) -> Dict:
        with open(os.path.join(self.path, "ontology.json"), "r") as fp:
            return json.load(fp)

    @cached_property
    def df_train(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.path, "balanced_train_segments.csv"),
            skiprows=2,
            sep=", ",
            engine="python",
        )
        if self.use_unbalanced:
            df = pd.concat(
                (
                    df,
                    pd.read_csv(
                        os.path.join(
                            self.path, "unbalanced_train_segments.csv"
                        ),
                        skiprows=2,
                        sep=", ",
                        engine="python",
                    ),
                )
            )
        return self.map_to_classes(df)

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.path, "eval_segments.csv"),
            skiprows=2,
            sep=", ",
            engine="python",
        )
        return self.map_to_classes(df)

    @cached_property
    def df_test(self) -> pd.DataFrame:
        return self.df_dev

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download AudioSet.

        The audio files must be downloaded manually from
        https://research.google.com/audioset/download.html.
        We do not implement
        the download of the audio files
        as this is a very time consuming process.
        Instead,
        we download only the metadata files
        and rely on the user
        to download the audio themselves.

        Args:
            path: Path to the directory to download the dataset to.
        """
        pass


if __name__ == "__main__":
    data = AudioSet(
        path="/home/trianand/mnt/databases/UAU/AudioSet2021/",
        metrics=["autrainer.metrics.MLF1Weighted"],
        tracking_metric="autrainer.metrics.MLF1Weighted",
        file_handler="autrainer.datasets.utils.AudioFileHandler",
        file_type="wav",
        features_subdir="",
        seed=0,
        index_column="foo",
        target_column="bar",
        batch_size=1,
    )
    batch = next(iter(data.train_loader))
    print(batch)

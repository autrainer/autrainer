"""Code for AudioSet dataset.
Utility functions to filter data
adapted from: https://github.com/audeering/audtorch/blob/master/audtorch/datasets/audio_set.py
Originally written by https://github.com/hagenw
"""

from functools import cached_property
import json
import logging
import os
from typing import Dict, List, Optional, Union

from audeer import flatten_list
from omegaconf import DictConfig
import pandas as pd

from autrainer.datasets.abstract_dataset import BaseMLClassificationDataset
from autrainer.transforms import SmartCompose


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
        features_subdir: Optional[str],
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        target_column: Optional[List[str]] = None,
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
                If None, defaults to the least common subset
                available in the data. Defaults to None.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
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
            threshold: Threshold for classification. Defaults to 0.5.
            use_unbalanced: Flag to allow the use of unbalanced set
                during training.
            include: List of categories to include. If set,
                only instances tagged with at least one
                category from the list will be included
                in the final dataset.
            exclude: List of categories to include. If set,
                all instances tagged with at least one
                category from the list will be excluded
                from the final dataset.
        """
        self.use_unbalanced = use_unbalanced
        self.include = include
        self.exclude = exclude
        self._log = logging.getLogger(__name__)
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
            threshold=threshold,
        )

    @property
    def audio_subdir(self) -> str:
        """Subdirectory containing audio data.

        Data assumed to be in root directory
        under the name of the respective partition
        which is prepended once the CSV file is loaded.
        Available partitions are (with original number of instances):
        - balanced_train_segments (n=22,160)
        - unbalanced_train_segments (n=2,041,789)
        - eval_segments (n=20,371)
        """
        return ""

    def _assert_target_column(self, allowed_columns: List[str]) -> None:
        """Override target column check.

        Allow `None` as a valid input.
        Override it to support all available tags
        in ontology, irrespective of whether
        they appear at all in the data or not.
        """
        if self.target_column is None:
            unique_categories = {x["name"] for x in self.ontology}
            train_categories = {
                x for y in self.df_train["categories"].values.tolist() for x in y
            }
            dev_categories = {
                x for y in self.df_dev["categories"].values.tolist() for x in y
            }
            self.target_column = list(
                (
                    train_categories.intersection(dev_categories).intersection(
                        unique_categories
                    )
                )
            )
            self._log.info(
                (
                    f"Training on {len(self.target_column)} classes:\n"
                    f"{self.target_column}"
                )
            )
        return super()._assert_target_column(allowed_columns)

    def _filename_and_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        r"""Return data frame with filenames and IDs.

        Args:
            df: data frame as read in from the CSV file.

        Results:
            data frame with columns `filename` and `ids`.

        """
        df = df.rename(columns={"positive_labels": "ids"})
        # Translate labels from "label1,label2" to [label1, label2]
        df["ids"] = [label.strip('"').split(",") for label in df["ids"]]
        # Insert filename
        df["filename"] = df["# YTID"] + ".wav"
        return df[["filename", "ids"]]

    def _add_parent_ids(self, child_ids: List[str]) -> List[str]:
        r"""Add all parent IDs to the list of given child IDs.

        Args:
            child_ids: child IDs.

        Return:
            list of child and parent IDs.

        """
        ids = child_ids
        for id in child_ids:
            ids += [x["id"] for x in self.ontology if id in x["child_ids"]]
        # Remove duplicates
        return list(set(ids))

    def _convert_ids_to_categories(self, ids: List[str]) -> List[str]:
        r"""Convert list of ids to sorted list of categories.

        Args:
            ids: list of IDs.

        Returns:
            list of sorted categories.

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
        return [cat for _, cat in sorted(zip(order, categories, strict=True))]

    def _filter_by_categories(
        self,
        df: pd.DataFrame,
        categories: List[str],
        exclude_mode: bool = False,
    ) -> pd.DataFrame:
        r"""Return data frame containing only specified categories.

        Args:
            df: data frame containing the columns `ids`.
            categories: list of categories to include or exclude.
            exclude_mode: if `False` the specified categories
                should be included in the data frame, otherwise excluded.
                Default: `False`.

        Returns:
            data frame containing only the desired categories.

        """
        ids = self._ids_for_categories(categories)
        if exclude_mode:
            # Remove rows that have an intersection of actual and desired IDs
            df = df.loc[
                df.apply(
                    lambda row: not set(row["ids"]) & set(ids),
                    axis=1,
                )
            ]
        else:
            # Include rows that have an intersection of actual and desired IDs
            df = df.loc[
                df.apply(
                    lambda row: bool(set(row["ids"]) & set(ids)),
                    axis=1,
                )
            ]
        return df.reset_index(drop=True)

    def _ids_for_categories(self, categories: List[str]) -> List[str]:
        r"""All IDs and child IDs for a given set of categories.

        Args:
            categories: list of categories.

        Returns:
            list of IDs.

        """
        ids = []
        category_ids = [x["id"] for x in self.ontology if x["name"] in categories]
        for category_id in category_ids:
            ids += self._subcategory_ids(category_id)
        # Remove duplicates
        return list(set(ids))

    def _subcategory_ids(self, parent_id: str) -> List[str]:
        r"""Recursively identify all IDs of a given category.

        Args:
            parent_id: ID of parent category

        Returns:
            list of all children IDs and the parent ID

        """
        id_list = [parent_id]
        child_ids = [x["child_ids"] for x in self.ontology if x["id"] == parent_id]
        child_ids = flatten_list(child_ids)
        # Add all subcategories
        for child_id in child_ids:
            id_list += self._subcategory_ids(child_id)
        return id_list

    def map_to_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map dataframe as loaded to `autrainer` compatible.

        Assigns all AudioSet IDs to category names.
        Recursively assigns parent nodes
        to every instance where a child is present.
        Optionally filters dataframes
        to include/exclude instances.
        Creates columns with final labels.

        Args:
            df: dataframe loaded from CSV.

        Returns:
            Adapted dataframe.
        """
        df = self._filename_and_ids(df)
        if self.include is not None:
            df = self._filter_by_categories(df, self.include)
        df["ids"] = df["ids"].map(self._add_parent_ids)
        if self.exclude is not None:
            df = self._filter_by_categories(df, self.exclude, exclude_mode=True)
        df["categories"] = df["ids"].map(self._convert_ids_to_categories)
        unique_categories = list({x["name"] for x in self.ontology})
        for c in unique_categories:
            df = pd.concat(
                (
                    df,
                    df["categories"]
                    .apply(lambda x: 1 if c in x else 0)  # noqa: B023
                    .to_frame(name=c),
                ),
                axis=1,
            )
        return df

    @cached_property
    def ontology(self) -> Dict:
        """Reads AudioSet ontology."""
        with open(os.path.join(self.path, "ontology.json"), "r") as fp:
            return json.load(fp)

    def _load_df(self, relative_path: str) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.path, f"{relative_path}.csv"),
            skiprows=2,
            sep=", ",
            engine="python",
        )
        df = self.map_to_classes(df)
        df["filename"] = df["filename"].apply(lambda x: os.path.join(relative_path, x))
        orig_len = len(df)
        df = df.loc[
            df["filename"].apply(
                lambda x: os.path.isfile(
                    os.path.join(
                        self.features_path,
                        self.features_subdir,
                        x.replace(".wav", f".{self.file_type}"),
                    )
                )
            )
        ]
        avail_len = len(df)
        if avail_len != orig_len:
            self._log.warning(
                f"Found only {avail_len} "
                f"from the {orig_len} original samples "
                f"included in the {relative_path} set of Audioset. "
            )
        return df

    @cached_property
    def df_train(self) -> pd.DataFrame:
        df = self._load_df("balanced_train_segments")
        if self.use_unbalanced:
            u_df = self._load_df("unbalanced_train_segments")
            df = pd.concat((df, u_df))
        return df

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        return self._load_df("eval_segments")

    @cached_property
    def df_test(self) -> pd.DataFrame:
        return self.df_dev

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download AudioSet.

        The data must be downloaded manually from
        https://research.google.com/audioset/download.html.
        We do not implement
        an automatic download
        as this is a very costly process
        that the user should handle
        outside of `autrainer`.

        Args:
            path: Path to the directory to download the dataset to.
        """
        pass

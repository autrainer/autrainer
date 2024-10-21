import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig
import pandas as pd

from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseClassificationDataset
from .utils import ZipDownloadManager


FILES = {
    # TUT Acoustic scenes 2016, Development dataset
    "TUT-acoustic-scenes-2016-development.audio.1.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.audio.1.zip?download=1",
    "TUT-acoustic-scenes-2016-development.audio.2.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.audio.2.zip?download=1",
    "TUT-acoustic-scenes-2016-development.audio.3.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.audio.3.zip?download=1",
    "TUT-acoustic-scenes-2016-development.audio.4.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.audio.4.zip?download=1",
    "TUT-acoustic-scenes-2016-development.audio.5.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.audio.5.zip?download=1",
    "TUT-acoustic-scenes-2016-development.audio.6.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.audio.6.zip?download=1",
    "TUT-acoustic-scenes-2016-development.audio.7.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.audio.7.zip?download=1",
    "TUT-acoustic-scenes-2016-development.audio.8.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.audio.8.zip?download=1",
    "TUT-acoustic-scenes-2016-development.error.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.error.zip?download=1",
    "TUT-acoustic-scenes-2016-development.meta.zip": "https://zenodo.org/records/45739/files/TUT-acoustic-scenes-2016-development.meta.zip?download=1",
    # TUT Acoustic scenes 2016, Evaluation dataset
    "TUT-acoustic-scenes-2016-evaluation.audio.1.zip": "https://zenodo.org/records/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.1.zip?download=1",
    "TUT-acoustic-scenes-2016-evaluation.audio.2.zip": "https://zenodo.org/records/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.2.zip?download=1",
    "TUT-acoustic-scenes-2016-evaluation.audio.3.zip": "https://zenodo.org/records/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.3.zip?download=1",
    "TUT-acoustic-scenes-2016-evaluation.meta.zip": "https://zenodo.org/records/165995/files/TUT-acoustic-scenes-2016-evaluation.meta.zip?download=1",
}


class DCASE2016Task1(BaseClassificationDataset):
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
        fold: int = 1,
    ) -> None:
        """TUT Acoustic scenes 2016 Task 1 (DCASE2016Task1) dataset.

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
            fold: Fold to use in [1, 2, 3, 4]. Defaults to 1.
        """
        self._assert_choice(fold, [1, 2, 3, 4], "fold")
        self.fold = fold
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
        return (
            pd.read_csv(os.path.join(self.path, f"fold{self.fold}_train.csv")),
            pd.read_csv(
                os.path.join(self.path, f"fold{self.fold}_evaluate.csv")
            ),
            pd.read_csv(os.path.join(self.path, "test.csv")),
        )

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the TUT Acoustic scenes 2016 Task 1 (DCASE2016Task1)
        dataset.

        For more information on the dataset and dataset split, see:
        https://dcase.community/challenge2016/task-acoustic-scene-classification

        Args:
            path: Path to the directory to download the dataset to.
        """
        out_path = os.path.join(path, "default")
        if os.path.isdir(out_path):
            return
        os.makedirs(out_path, exist_ok=True)

        # download and extract files
        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(
            check_exist=[
                "TUT-acoustic-scenes-2016-development",
                "TUT-acoustic-scenes-2016-evaluation",
            ]
        )
        dl_manager.extract(
            check_exist=[
                "TUT-acoustic-scenes-2016-development",
                "TUT-acoustic-scenes-2016-evaluation",
            ]
        )

        # move audio files to the same directory
        dev_path = os.path.join(path, "TUT-acoustic-scenes-2016-development")
        eval_path = os.path.join(path, "TUT-acoustic-scenes-2016-evaluation")
        for file in os.listdir(os.path.join(dev_path, "audio")):
            shutil.move(os.path.join(dev_path, "audio", file), out_path)
        for file in os.listdir(os.path.join(eval_path, "audio")):
            shutil.move(os.path.join(eval_path, "audio", file), out_path)

        # load folds
        fold_names = [
            f
            for f in os.listdir(os.path.join(dev_path, "evaluation_setup"))
            if f.endswith(".txt") and "_test" not in f
        ]
        folds = [
            pd.read_csv(
                os.path.join(dev_path, "evaluation_setup", f),
                sep="\t",
                header=None,
                names=["filename", "scene_label"],
            )
            for f in fold_names
        ]
        fold_names.append("test.txt")
        folds.append(
            pd.read_csv(
                os.path.join(eval_path, "evaluation_setup", "evaluate.txt"),
                sep="\t",
                header=None,
                names=["filename", "scene_label", "full_filename"],
            )
        )

        # remove erroneous files
        erroneous = pd.read_csv(
            os.path.join(dev_path, "error.txt"),
            sep="\t",
            header=None,
            names=["filename", "begin", "end", "error"],
        )
        folds = [
            fold[~fold["filename"].isin(erroneous["filename"])]
            for fold in folds
        ]

        # remove audio from path
        for fold in folds:
            fold.loc[:, "filename"] = fold["filename"].apply(os.path.basename)

        # rename test files to match the development files
        folds[-1].loc[:, "full_filename"] = folds[-1]["full_filename"].apply(
            os.path.basename
        )
        for short_name, full_name in folds[-1][
            ["filename", "full_filename"]
        ].values:
            if not os.path.isfile(os.path.join(out_path, short_name)):
                continue
            shutil.move(
                os.path.join(out_path, short_name),
                os.path.join(out_path, full_name),
            )
        folds[-1]["filename"] = folds[-1]["full_filename"]
        folds[-1].drop(columns=["full_filename"], inplace=True)

        # save folds
        for filename, fold in zip(fold_names, folds):
            fold.to_csv(
                os.path.join(path, Path(filename).with_suffix(".csv")),
                index=False,
            )

        # remove unnecessary files
        shutil.rmtree(dev_path)
        shutil.rmtree(eval_path)

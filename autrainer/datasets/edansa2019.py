import os
import shutil
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
import pandas as pd

from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseMLClassificationDataset
from .utils import ZipDownloadManager


FILES = {
    "EDANSA-2019.zip": "https://zenodo.org/records/6824272/files/EDANSA-2019.zip?download=1"
}


class EDANSA2019(BaseMLClassificationDataset):
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
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        """EDANSA 2019 dataset.

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
            threshold: Threshold for classification. Defaults to 0.5.
        """
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
            threshold=threshold,
        )
        self._assert_target_column(allowed_columns=self.df_train.columns[9:])

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the EDANSA 2019 dataset.

        For more information on the dataset, see:
        https://zenodo.org/doi/10.5281/zenodo.6824271

        Args:
            path: Path to the directory to download the dataset to.
        """
        out_path = os.path.join(path, "default")
        if os.path.isdir(out_path):
            return
        os.makedirs(out_path, exist_ok=True)

        # download and extract files
        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(check_exist=["EDANSA-2019"])
        dl_manager.extract(check_exist=["EDANSA-2019"])

        # move audio files
        for item in os.listdir(os.path.join(path, "EDANSA-2019", "data")):
            shutil.move(
                os.path.join(path, "EDANSA-2019", "data", item),
                os.path.join(out_path, item),
            )

        # process dataframes
        df = pd.read_csv(os.path.join(path, "EDANSA-2019", "labels.csv"))
        target_columns = df.columns[9:]
        for col in target_columns:
            df[col] = df[col].astype(float)
        df_train = df.loc[df["set"] == "train"]
        df_dev = df.loc[df["set"] == "valid"]
        df_test = df.loc[df["set"] == "test"]
        df_train.to_csv(os.path.join(path, "train.csv"), index=False)
        df_dev.to_csv(os.path.join(path, "dev.csv"), index=False)
        df_test.to_csv(os.path.join(path, "test.csv"), index=False)

        # remove unnecessary files
        shutil.rmtree(os.path.join(path, "EDANSA-2019"))

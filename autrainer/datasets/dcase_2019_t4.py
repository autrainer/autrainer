import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig
import pandas as pd

from autrainer.datasets.utils.dataset_wrapper import SegmentedDatasetWrapper
from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseSEDDataset
from .utils import ZipDownloadManager


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
        index_column: str = "filename",
        target_column: str = "segment_events",
        file_type: str = "npy",
        file_handler: Union[str, DictConfig, Dict] = "numpy",
        batch_size: int = 24,
        inference_batch_size: Optional[int] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
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
            min_event_length=0.25,
            min_event_gap=0.15,
        )

    def load_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation and test dataframes.
        
        Returns:
            Tuple containing (train_df, val_df, test_df)
        """
        base_path = Path(self.path)
        return (
            pd.read_csv(base_path / "synthetic_train.csv"),
            pd.read_csv(base_path / "synthetic_val.csv"),
            pd.read_csv(base_path / "public_test.csv")
        )
    
    def _split_train_dataset(
        self,
        df: pd.DataFrame,
        val_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train and validation sets.
        
        Args:
            df: DataFrame to split
            val_size: Fraction of data to use for validation
            
        Returns:
            Tuple containing (train_df, val_df, None)
        """
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(df.index)
        val_split = int(len(indices) * (1 - val_size))
        train_indices = indices[:val_split]
        val_indices = indices[val_split:]
        return (
            df.loc[train_indices].copy(),
            df.loc[val_indices].copy(),
            None
        )

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the DCASE 2019 Task 4 dataset.

        For more information on the dataset and dataset split, see:
        https://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments

        Args:
            path: Path to download the dataset to
            create_windows: If True, creates fixed windows during download
        """
        out_path = os.path.join(path, "default")
        if os.path.isdir(out_path):
            return
        os.makedirs(out_path, exist_ok=True)

        # download and extract files
        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(
            check_exist=[
                "Synthetic_dataset", 
                "DESED_public_eval"
            ]
        )
        dl_manager.extract(
            check_exist=[
                "Synthetic_dataset", 
                "DESED_public_eval"
            ]
        )

        # move audio files to the same directory
        synth_dev_path = os.path.join(path, "audio") # synthetic development set
        pub_eval_path = os.path.join(path, "dataset") # public evaluation set

        for file in os.listdir(os.path.join(synth_dev_path, "train", "synthetic")):
            if file.endswith(".wav"):
                shutil.move(os.path.join(synth_dev_path, "train", "synthetic", file), out_path)
        for file in os.listdir(os.path.join(pub_eval_path, "audio", "eval", "public")):
            if file.endswith(".wav"):
                shutil.move(os.path.join(pub_eval_path, "audio", "eval", "public", file), out_path)
        shutil.copy2(
            os.path.join(pub_eval_path, "metadata", "eval", "public.tsv"),
            os.path.join(path, "public_test.csv")
        )

        def process_dataset(csv_path: str, **kwargs) -> pd.DataFrame:
            df = pd.read_csv(csv_path, sep="\t")
            return SegmentedDatasetWrapper.create_fixed_windows(
                df, path=out_path, window_size=DURATIONS["min_dur_event"],
                min_event_length=DURATIONS["min_dur_event"],
                event_list=EVENTS, seq2seq=True, **kwargs
            )

        train_df = process_dataset(os.path.join(path, "synthetic_dataset.csv"))
        pub_eval_df = process_dataset(os.path.join(path, "public_test.csv"))
        pub_eval_df.to_csv(os.path.join(path, "public_test.csv"), index=False)

        # TODO: adapt official train/val split with synthetic + real data
        val_size = 0.2
        rng = np.random.default_rng(42)
        indices = rng.permutation(train_df.index)
        val_split = int(len(indices) * (1 - val_size))
        train_indices = indices[:val_split]
        val_indices = indices[val_split:]
        train_df.loc[train_indices].to_csv(
            os.path.join(path, "synthetic_train.csv"), index=False)
        train_df.loc[val_indices].to_csv(
            os.path.join(path, "synthetic_val.csv"), index=False)
        
        def remove_if_exists(path: str) -> None:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        for temp_dir in [synth_dev_path, pub_eval_path]:
            remove_if_exists(temp_dir)
        for license_file in ["._license_public_eval.tsv", "license_public_eval.tsv"]:
            remove_if_exists(os.path.join(path, license_file))
        for archive in ["DESED_public_eval.tar.gz", "Synthetic_dataset.zip"]:
            remove_if_exists(os.path.join(path, archive))
        remove_if_exists(os.path.join(path, "synthetic_dataset.csv"))
        remove_if_exists(os.path.join(path, "__MACOSX"))
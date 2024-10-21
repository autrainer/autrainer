import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig
import pandas as pd
from torchaudio import datasets

from autrainer.transforms import SmartCompose

from .abstract_dataset import BaseClassificationDataset


class SpeechCommands(BaseClassificationDataset):
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
    ) -> None:
        """Speech Commands (v0.02) dataset.

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
        )

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the Speech Commands (v0.02) dataset from torchaudio.

        For more information on the dataset, see:
        https://doi.org/10.48550/arXiv.1804.03209

        Args:
            path: Path to the directory to download the dataset to.
        """
        if os.path.isdir(os.path.join(path, "default")):
            return

        _version = "speech_commands_v0.02"
        subsets = ["training", "validation", "testing"]
        df_names = ["train", "dev", "test"]
        os.makedirs(path, exist_ok=True)
        datasets.SPEECHCOMMANDS(
            root=path,
            download=True,
            folder_in_archive=_version,
        )

        for subset, df_name in zip(subsets, df_names):
            dataset = datasets.SPEECHCOMMANDS(
                root=path,
                subset=subset,
                download=False,
                folder_in_archive=_version,
            )
            metadata = []
            for n in range(len(dataset)):
                meta = dataset.get_metadata(n)
                metadata.append(
                    {
                        "path": Path(meta[0]).relative_to(_version),
                        "sample_rate": meta[1],
                        "label": meta[2],
                        "speaker_id": meta[3],
                        "utterance_number": meta[4],
                    }
                )
            pd.DataFrame(metadata).to_csv(
                os.path.join(path, f"{df_name}.csv"),
                index=False,
            )

        shutil.move(
            os.path.join(path, _version, _version),
            os.path.join(path, "default"),
        )
        shutil.rmtree(os.path.join(path, _version))

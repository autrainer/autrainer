import os
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig
import pandas as pd
import torch

from autrainer.transforms import AbstractTransform, SmartCompose

from .abstract_dataset import BaseClassificationDataset


class Standardizer(AbstractTransform):
    def __init__(self, mean: List[float], std: List[float]) -> None:
        super().__init__(order=-100)
        self.mean = mean
        self.std = std
        self._mean = torch.Tensor(mean)
        self._std = torch.Tensor(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std)

    def decode(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x * self._std + self._mean


class AIBO(BaseClassificationDataset):
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
        standardize: bool = False,
        aibo_task: str = "2cl",
    ) -> None:
        """FAU AIBO dataset.

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
            standardize: Whether to standardize the data. Defaults to False.
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
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )
        self.standardize = standardize
        if self.standardize:
            train_data = torch.cat([x for x, *_ in self.train_dataset])
            print(train_data.mean(0).shape, train_data.std(0).shape)
            standardizer = Standardizer(
                mean=train_data.mean(0).tolist(),
                std=train_data.std(0).tolist(),
            )

            self.train_transform += standardizer
            self.dev_transform += standardizer
            self.test_transform += standardizer

            self.train_dataset = self._init_dataset(
                self.df_train, self.train_transform
            )

    def load_dataframes(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            pd.read_csv(
                os.path.join(self.path, f"train_{self.aibo_task}.csv")
            ),
            pd.read_csv(os.path.join(self.path, f"dev_{self.aibo_task}.csv")),
            pd.read_csv(os.path.join(self.path, f"test_{self.aibo_task}.csv")),
        )

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

        Produces the following splits for both tasks (2cl and 5cl):

        - `train_{task}.csv`: Training split of all speakers of the
          Ohm-Gymnasium with the exception of the last two speakers.
        - `dev_{task}.csv`: Development split of the last two speakers of the
          Ohm-Gymnasium.
        - `test_{task}.csv`: Test split of all speakers of the
          Montessori-Schule.

        For more information on the dataset and dataset split, see:
        https://doi.org/10.1109/ICME51207.2021.9428217

        Args:
            path: Path to the directory to download the dataset to.
        """
        if os.path.isfile(os.path.join(path, "train_2cl.csv")):
            return
        if not os.path.isdir(os.path.join(path, "default")):
            raise ValueError(
                f"Directory 'default' does not exist in '{path}'."
            )
        if not os.path.isfile(
            os.path.join(path, "chunk_labels_2cl_corpus.txt")
            or os.path.isfile(
                os.path.join(path, "chunk_labels_5cl_corpus.txt")
            )
        ):
            raise ValueError(
                f"File 'chunk_labels_2cl_corpus.txt' or "
                f"'chunk_labels_5cl_corpus.txt' does not exist in '{path}'."
            )

        for task in ["2cl", "5cl"]:
            df = pd.read_csv(
                os.path.join(path, f"chunk_labels_{task}_corpus.txt"),
                header=None,
                sep=" ",
            )
            df = df.rename(columns={0: "id", 1: "class", 2: "conf"})
            df["file"] = df["id"].apply(lambda x: x + ".wav")
            df["school"] = df["id"].apply(lambda x: x.split("_")[0])
            df["speaker"] = df["id"].apply(lambda x: x.split("_")[1])
            df = df.set_index("file")
            df_test = df.loc[df["school"] == "Mont"]
            df_train_dev = df.loc[df["school"] == "Ohm"]
            speakers = sorted(df_train_dev["speaker"].unique())
            df_train = df_train_dev.loc[
                df_train_dev["speaker"].isin(speakers[:-2])
            ]
            df_dev = df_train_dev.loc[
                df_train_dev["speaker"].isin(speakers[-2:])
            ]

            df_train.to_csv(os.path.join(path, f"train_{task}.csv"))
            df_dev.to_csv(os.path.join(path, f"dev_{task}.csv"))
            df_test.to_csv(os.path.join(path, f"test_{task}.csv"))

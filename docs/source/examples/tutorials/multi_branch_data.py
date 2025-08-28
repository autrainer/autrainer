from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch

from autrainer.core.structs import AbstractDataBatch, AbstractDataItem
from autrainer.datasets.toy_dataset import ToyDataset, ToyDatasetWrapper
from autrainer.transforms import SmartCompose


@dataclass
class DataItemMultiBranch(AbstractDataItem):
    features: torch.Tensor
    meta: torch.Tensor
    target: int
    index: int


@dataclass
class DataBatchMulti(AbstractDataBatch[DataItemMultiBranch]):
    """Data batch class for a batch of data samples.

    Args:
        features: Tensor of input features.
        meta: Tensor of input support features.
        target: Tensor of target values for the input features.
        index: Tensor of indices for the data samples.
    """

    features: torch.Tensor
    meta: torch.Tensor
    target: torch.Tensor
    index: torch.Tensor

    def to(self, device: torch.device) -> None:
        self.features = self.features.to(device)
        self.meta = self.meta.to(device)
        self.target = self.target.to(device)


class ToyDatasetMultiBranch(ToyDatasetWrapper):
    def __getitem__(self, index: int) -> DataItemMultiBranch:
        data = super().__getitem__(index)
        return DataItemMultiBranch(
            features=data.features,
            target=data.target,
            index=data.index,
            meta=data.features,
        )


class ToyMultiBranchData(ToyDataset):
    def _init_dataset(
        self,
        df: pd.DataFrame,
        transform: SmartCompose,
    ) -> ToyDatasetMultiBranch:
        return ToyDatasetMultiBranch(
            df=df,
            target_column=self.target_column,
            feature_shape=self.feature_shape,
            dtype=self.dtype,
            generator=self._generator,
            transform=transform,
            target_transform=self.target_transform,
        )

    @property
    def default_collate_fn(self) -> Callable:
        return DataBatchMulti.collate

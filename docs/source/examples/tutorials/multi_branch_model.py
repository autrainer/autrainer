from dataclasses import dataclass

import pandas as pd
import torch

from autrainer.datasets.toy_dataset import ToyDataset, ToyDatasetWrapper
from autrainer.datasets.utils.data_struct import AbstractDataBatch
from autrainer.models import AbstractModel
from autrainer.transforms import SmartCompose


@dataclass
class DataItemMultiBranch:
    features: torch.Tensor
    target: int
    index: int
    meta: torch.Tensor


class ToyDatasetMultiBranch(ToyDatasetWrapper):
    def __getitem__(self, index: int) -> DataItemMultiBranch:
        data = super().__getitem__(index)
        return DataItemMultiBranch(
            features=data.features,
            target=data.target,
            index=data.index,
            meta=data.features,
        )


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
    def default_collate_fn(self):
        return DataBatchMulti.collate


class ToyMultiBranchModel(AbstractModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim * 2, self.output_dim)

    def embeddings(
        self, features: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        return torch.concat(
            [self.linear1(features), self.linear2(meta)], axis=1
        )

    def forward(
        self, features: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        return self.out(self.embeddings(features=features, meta=meta))

from dataclasses import dataclass

import torch

from .abstract_data_struct import AbstractDataBatch, AbstractDataItem


@dataclass
class DataItem(AbstractDataItem):
    """Data item for a single sample.

    Args:
        features: Tensor of input features.
        target: Target value for the input features.
        index: Index of the data sample.
    """


@dataclass
class DataBatch(AbstractDataBatch[DataItem]):
    """Data batch for a batch of samples.

    Args:
        features: Tensor of input features.
        target: Tensor of target values for the input features.
        index: Tensor of indices for the data samples.
    """

    def to(self, device: torch.device, **kwargs: dict) -> None:
        self.features = self.features.to(device, **kwargs)
        self.target = self.target.to(device, **kwargs)

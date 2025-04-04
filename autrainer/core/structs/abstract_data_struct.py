from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Mapping, Type, TypeVar, Union

import numpy as np
import torch


AbstractItemType = TypeVar("ItemType", bound="AbstractDataItem")


@dataclass
class AbstractDataItem(ABC):
    """Abstract data item class for a single sample.

    Args:
        features: Tensor of input features.
        target: Target value for the input features.
        index: Index of the data sample.
    """

    features: torch.Tensor
    target: Union[int, float, List[int], List[float], np.ndarray]
    index: int


@dataclass
class AbstractDataBatch(ABC, Generic[AbstractItemType]):
    """Abstract data batch class for a batch of samples.

    Args:
        features: Tensor of input features.
        target: Tensor of target values for the input features.
        index: Tensor of indices for the data samples.
    """

    features: torch.Tensor
    target: torch.Tensor
    index: torch.Tensor

    @abstractmethod
    def to(self, device: torch.device, **kwargs: dict) -> None:
        """Move the features, target, and additional data to a device.

        Args:
            device: Device to move the data to.
            kwargs: Additional keyword arguments passed to the `to` method.
        """

    @classmethod
    def collate(
        cls: Type["AbstractDataBatch[AbstractItemType]"],
        items: List[AbstractItemType],
    ) -> "AbstractDataBatch[AbstractItemType]":
        """Collate a list of data items into a data batch.

        Args:
            items: List of data items to collate.

        Returns:
            Collated data batch.
        """
        if not isinstance(items[0], Mapping):
            items = [vars(item) for item in items]
        first = items[0]
        batch = {}

        for k, v in first.items():
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in items])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(np.stack([f[k] for f in items]))
            else:
                batch[k] = torch.tensor([f[k] for f in items])

        return cls(**batch)

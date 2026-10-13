from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import torch


AbstractItemType = TypeVar("AbstractItemType", bound="AbstractDataItem")


@dataclass
class AbstractDataItem(ABC):
    """Abstract data item class for a single sample.

    Args:
        features: Tensor of input features.
        target: Target value for the input features.
        index: Index of the data sample.
    """

    features: torch.Tensor
    target: int | float | list[int] | list[float] | np.ndarray
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
        cls: type["AbstractDataBatch[AbstractItemType]"],
        items: list[AbstractItemType],
    ) -> "AbstractDataBatch[AbstractItemType]":
        """Collate a list of data items into a data batch.

        Args:
            items: List of data items to collate.

        Returns:
            Collated data batch.
        """
        if not items:
            raise RuntimeError("Collate requires at least one item.")

        fn = dict if isinstance(items[0], Mapping) else vars
        mapping: list[dict[str, Any]] = [fn(item) for item in items]  # ty: ignore[no-matching-overload]

        batch = {}
        for k, v in mapping[0].items():
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in mapping])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(np.stack([f[k] for f in mapping]))
            else:
                batch[k] = torch.tensor([f[k] for f in mapping])

        return cls(**batch)

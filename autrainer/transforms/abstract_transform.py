from abc import ABC, abstractmethod
from typing import Optional

import audobject
import torch


class AbstractTransform(ABC, audobject.Object):
    def __init__(self, order: int = 0, **kwargs) -> None:
        """Abstract class for a transform.

        Args:
            order: The order of the transform in the pipeline. Larger means
                later in the pipeline. If multiple transforms have the same
                order, they are applied in the order they were added to the
                pipeline. Defaults to 0.
            kwargs: Additional keyword arguments to store in the object.
        """
        super().__init__(**kwargs)
        self.order = order
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        args_dict = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            args_dict[k] = v
        args_repr = ", ".join(f"{k}={v!r}" for k, v in args_dict.items())
        return self.__class__.__name__ + f"({args_repr})"

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        index: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply the transform to the input tensor.

        Args:
            x: The input tensor.
            index: The index of the input tensor in the dataset.
                Defaults to None.

        Returns:
            The transformed tensor.
        """

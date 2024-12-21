import importlib
import inspect
from typing import TYPE_CHECKING

import torch

from .abstract_transform import AbstractTransform


if TYPE_CHECKING:
    from autrainer.datasets.utils import DatasetWrapper


class GlobalTransform(AbstractTransform):
    def __init__(
        self,
        transform: str,
        skip_augmentations: bool = True,
        resolve_by_position: bool = True,
        **kwargs,
    ) -> None:
        """Wrapper to apply global transforms to a dataset.

        Args:
            transform: Fully qualified name of the transform to apply.
            skip_augmentations: Whether to skip possible augmentations in the
                transformation pipeline when resolving the global transform.
                Defaults to True.
            resolve_by_position: Whether to positionally resolve the global
                transform in the transformation pipeline and only apply
                preceding transforms. Defaults to True.

        Raises:
            ValueError: If the transform is not a fully qualified name.
            ValueError: If the transform does not have a class method
                `from_global` to be instantiated as a global transform.
        """
        if "." not in transform:
            raise ValueError("The transform must be a fully qualified name.")
        module, cls = transform.rsplit(".", 1)
        cls = getattr(importlib.import_module(module), cls)
        _method = inspect.getattr_static(cls, "from_global")
        if not isinstance(_method, classmethod):
            raise ValueError(
                f"Transform '{cls.__name__}' requires a class method "
                "`from_global` to be instantiated as a global transform."
            )

        order = inspect.signature(cls).parameters.get("order")
        if order and not isinstance(order.default, inspect._empty):
            order = order.default
        else:
            order = 0
        super().__init__(order=order)
        self.transform = transform
        self.skip_augmentations = skip_augmentations
        self.resolve_by_position = resolve_by_position
        self.kwargs = kwargs
        self._cls = cls

    def setup(self, data: "DatasetWrapper") -> None:
        """Setup the global transform with the dataset.

        Args:
            data: Dataset to setup the transform with.
        """
        transform = self._cls.from_global(data, **self.kwargs)
        self.__class__ = transform.__class__
        self.__dict__ = transform.__dict__

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data

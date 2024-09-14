from typing import TYPE_CHECKING, Callable, List, Optional, Union

import audobject
import torch
from torchvision import transforms as T

from .abstract_transform import AbstractTransform


if TYPE_CHECKING:
    from autrainer.datasets import AbstractDataset  # pragma: no cover


class SmartCompose(T.Compose, audobject.Object):
    def __init__(
        self,
        transforms: List[Union[T.Compose, AbstractTransform, Callable]],
        **kwargs,
    ) -> None:
        """SmartCompose wrapper for torchvision.transforms.Compose, which
        allows for simple composition of transforms by adding them together.
        Transforms are automatically sorted by their order attribute if present.

        Additionally, SmartCompose allows for the specification of a
        collate function to be used in a DataLoader, the last collate function
        specified will be used.

        Args:
            transforms: List of transforms to compose.
            **kwargs: Additional keyword arguments to pass to
                torchvision.transforms.Compose.
        """
        super().__init__(transforms, **kwargs)
        self._sort()

    def __add__(
        self,
        other: Union[T.Compose, AbstractTransform, Callable, List[Callable]],
    ) -> "SmartCompose":
        """Add another transform to the composition. Transforms are
        automatically sorted by their order attribute if present.

        Args:
            other: Transform to add to the composition.

        Raises:
            TypeError: If the addition is not a valid type.
                Supported types are 'torchvision.transforms.Compose',
                'AbstractTransform', 'Callable', or 'List[Callable]'.

        Returns:
            New SmartCompose object with the added transform.
        """
        if other is None:
            return self
        elif isinstance(other, T.Compose):
            return SmartCompose(
                self.transforms + other.transforms,
            )
        elif isinstance(other, (AbstractTransform, Callable)):
            return SmartCompose(self.transforms + [other])
        elif isinstance(other, List):
            return SmartCompose(self.transforms + other)
        else:
            raise TypeError(
                "SmartCompose addition must be a "
                "'torchvision.transforms.Compose', 'AbstractTransform', "
                f"'Callable', or 'List[Callable]' got '{type(other)}'."
            )

    def _sort(self):
        """Sort the transforms by their order attribute if present."""
        self.transforms.sort(key=lambda x: getattr(x, "order", 0))

    def get_collate_fn(self, data: "AbstractDataset") -> Optional[Callable]:
        """Get the collate function. If no collate function is present in
        the transforms, None is returned.
        If multiple collate functions are present, the last one is used.

        Args:
            data: Dataset to get the collate function for.

        Returns:
            Collate function.
        """
        collate_fn = None
        for t in self.transforms:
            if fn := getattr(t, "get_collate_fn", None):
                collate_fn = fn
        if collate_fn is not None:
            return collate_fn(data)

    def __call__(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """Apply the transforms to the input tensor.

        Args:
            x: Input tensor.
            index: Dataset index of the input tensor.

        Returns:
            Transformed tensor.
        """
        for t in self.transforms:
            if "index" in t.__call__.__annotations__.keys():
                x = t(x, index=index)
            else:
                x = t(x)
        return x

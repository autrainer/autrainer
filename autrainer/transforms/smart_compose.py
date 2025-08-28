from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import audobject
from torchvision import transforms as T

from autrainer.core.structs import AbstractDataItem

from .abstract_transform import AbstractTransform


if TYPE_CHECKING:  # pragma: no cover
    from autrainer.datasets import AbstractDataset


class SmartCompose(T.Compose, audobject.Object):
    def __init__(
        self,
        transforms: List[AbstractTransform],
        **kwargs: Dict[str, Any],
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
        other: Optional[
            Union["SmartCompose", AbstractTransform, List[AbstractTransform]]
        ],
    ) -> "SmartCompose":
        """Add another transform to the composition. Transforms are
        automatically sorted by their order attribute if present.

        Args:
            other: Transform to add to the composition.

        Raises:
            TypeError: If the addition is not a valid type.
                Supported types are AbstractTransform, SmartCompose,
                and list of AbstractTransform or SmartCompose.

        Returns:
            New SmartCompose object with the added transform.
        """
        if other is None:
            return self
        if isinstance(other, AbstractTransform):
            return SmartCompose(self.transforms + [other])
        if isinstance(other, SmartCompose):
            return SmartCompose(self.transforms + other.transforms)
        if isinstance(other, list):
            t = SmartCompose([])
            for o in other:
                if isinstance(o, AbstractTransform):
                    t.transforms.append(o)
                elif isinstance(o, SmartCompose):
                    t.transforms += o.transforms
            if len(t.transforms):
                return SmartCompose(self.transforms + t.transforms)
        raise TypeError(
            f"Unsupported type for addition: {type(other)}. "
            "Supported types are AbstractTransform, SmartCompose, "
            "and list of AbstractTransform or SmartCompose."
        )

    def _sort(self) -> None:
        """Sort the transforms by their order attribute if present."""
        self.transforms.sort(key=lambda x: getattr(x, "order", 0))

    def get_collate_fn(self, data: "AbstractDataset") -> Callable:
        """Get the collate function. If no collate function is present in
        the transforms, the dataset default is returned.
        If multiple collate functions are present, the last one is used.

        Args:
            data: Dataset to get the collate function for.
                Includes a ``default_collate_fn``.

        Returns:
            Collate function.
        """
        default_fn = data.default_collate_fn
        collate_fn = None
        for t in self.transforms:
            if fn := getattr(t, "get_collate_fn", None):
                collate_fn = fn
        if collate_fn is not None:
            return collate_fn(data, default_fn)
        return default_fn

    def setup(self, data: "AbstractDataset") -> "SmartCompose":
        for t in self.transforms:
            if hasattr(t, "setup"):
                t.setup(data)
        return self

    def offset_generator_seed(self, offset: int) -> None:
        """Offset the generator seed for transforms that use random
        number generators. Useful for ensuring reproducibility and
        randomness of augmentations when using multiple workers.

        Args:
            offset: Offset to add to the generator seed. Usually the
                worker index.
        """
        for t in self.transforms:
            if hasattr(t, "offset_generator_seed"):
                t.offset_generator_seed(offset)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        """Apply the transform to the input data item.

        Args:
            item: The input data item to transform.

        Returns:
            The transformed data item.
        """
        for t in self.transforms:
            item = t(item)
        return item

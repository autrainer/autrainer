from copy import deepcopy
from typing import Dict, List, Optional

import audobject

import autrainer
from autrainer.core.structs import AbstractDataItem

from .abstract_augmentation import AbstractAugmentation
from .utils import convert_shorthand


class Sequential(AbstractAugmentation, audobject.Object):
    def __init__(
        self,
        sequence: List[Dict],
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Create a fixed sequence of augmentations.

        The order of the augmentations in the list is not considered and is
        placed with respect to the order of the sequence augmentation itself.
        This means that the sequence of augmentations is applied in the order
        they are defined in the list and not disrupted by any other transform.

        Augmentations in the list must not have a collate function.

        Args:
            sequence: A list of (shorthand syntax) dictionaries defining the
                augmentation sequence.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)

        self.sequence = sequence
        self.augmentation_sequence = []

        for aug in deepcopy(self.sequence):
            aug = convert_shorthand(aug)
            aug_name = next(iter(aug.keys()))
            if aug[aug_name].get("generator_seed") is None:
                aug[aug_name]["generator_seed"] = self.generator_seed
            self.augmentation_sequence.append(aug)

        self.augmentation_sequence: List[AbstractAugmentation] = [
            autrainer.instantiate_shorthand(aug, instance_of=AbstractAugmentation)
            for aug in self.augmentation_sequence
        ]
        for aug in self.augmentation_sequence:
            if hasattr(aug, "get_collate_fn"):
                raise ValueError(
                    "Choice augmentations must not have a collate function."
                )

    def offset_generator_seed(self, offset: int) -> None:
        super().offset_generator_seed(offset)
        for aug in self.augmentation_sequence:
            aug.offset_generator_seed(offset)

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        """Apply all augmentations in sequence to the input tensor.

        Args:
            item: The input data item.

        Returns:
            The augmented item.
        """
        for aug in self.augmentation_sequence:
            item = aug(item)
        return item

    @property
    def _deterministic(self) -> bool:
        """Return True if the augmentation is deterministic, False otherwise."""
        return all(
            getattr(aug, "_deterministic", True)  # true if not set
            for aug in self.augmentation_sequence
        )

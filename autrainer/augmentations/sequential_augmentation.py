from typing import Dict, List, Optional

import audobject
import torch

import autrainer

from .abstract_augmentation import AbstractAugmentation


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

        for aug in self.sequence:
            if isinstance(aug, str):
                self.augmentation_sequence.append(
                    {aug: {"generator_seed": self.generator_seed}}
                )
            else:
                aug_name = next(iter(aug.keys()))
                if aug[aug_name].get("generator_seed") is None:
                    aug[aug_name]["generator_seed"] = self.generator_seed
                self.augmentation_sequence.append(aug)

        self.augmentation_sequence: List[AbstractAugmentation] = [
            autrainer.instantiate_shorthand(
                aug,
                instance_of=AbstractAugmentation,
            )
            for aug in self.sequence
        ]
        for aug in self.augmentation_sequence:
            if hasattr(aug, "get_collate_fn"):
                raise ValueError(
                    "Choice augmentations must not have a collate function."
                )

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        """Apply all augmentations in sequence to the input tensor.

        Args:
            x: The input tensor.
            index: The index of the input tensor in the dataset.
                Defaults to None.

        Returns:
            The augmented tensor.
        """
        for aug in self.augmentation_sequence:
            x = aug(x, index)
        return x

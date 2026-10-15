from copy import deepcopy
from typing import Any

import audobject
from omegaconf import DictConfig
import torch

import autrainer
from autrainer.core.structs import AbstractDataItem

from .abstract_augmentation import AbstractAugmentation
from .utils import convert_shorthand


class Choice(AbstractAugmentation, audobject.Object):
    def __init__(
        self,
        choices: list[str | DictConfig | dict[str, Any]],
        weights: list[float] | None = None,
        order: int = 0,
        p: float = 1.0,
        generator_seed: int | None = None,
    ) -> None:
        """Choose one augmentation from a list of augmentations with a given
        probability.

        The order of the augmentations in the list is not considered and is placed with
        respect to the order of the choice augmentation itself.

        Augmentations in the list must not have a collate function.

        Args:
            choices: A list of (shorthand syntax) dictionaries defining the
                augmentations to choose from.
            weights: A list of weights for each choice. If None, all augmentations are
                assigned equal weights. Defaults to None.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number generator
                drawing the probability. If None, the generator is not seeded. Defaults
                to None.

        Raises:
            ValueError: If choices and weights have different lengths.
            ValueError: If any augmentation has a collate function.
        """
        super().__init__(order, p, generator_seed)
        weights = weights or [1.0] * len(choices)
        if len(choices) != len(weights):
            raise ValueError("Choices and weights must have the same length.")

        self.weights = [x / sum(weights) for x in weights]
        self.choices = choices
        converted = []

        for aug in deepcopy(self.choices):
            aug = convert_shorthand(aug)
            aug_name = next(iter(aug.keys()))
            if aug[aug_name].get("generator_seed") is None:
                aug[aug_name]["generator_seed"] = self.generator_seed
            converted.append(aug)

        self.augmentation_choices: list[AbstractAugmentation] = [
            autrainer.instantiate_shorthand(choice, instance_of=AbstractAugmentation)
            for choice in converted
        ]
        for aug in self.augmentation_choices:
            if hasattr(aug, "get_collate_fn"):
                raise ValueError(
                    "Choice augmentations must not have a collate function."
                )

    def offset_generator_seed(self, offset: int) -> None:
        super().offset_generator_seed(offset)
        for aug in self.augmentation_choices:
            aug.offset_generator_seed(offset)

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        """Choose one augmentation from the list of augmentations based on the given
        weights.

        Args:
            item: The input data item.

        Returns:
            The augmented item.
        """
        weights = torch.Tensor(self.weights)
        choice = int(torch.multinomial(weights, 1, generator=self.g).item())
        return self.augmentation_choices[choice](item)

    @property
    def _deterministic(self) -> bool:
        """Return True if the augmentation is deterministic, False otherwise."""
        return all(
            getattr(aug, "_deterministic", True)  # true if not set
            for aug in self.augmentation_choices
        )

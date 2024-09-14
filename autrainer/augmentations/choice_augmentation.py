from typing import Dict, List, Optional

import audobject
import torch

import autrainer

from .abstract_augmentation import AbstractAugmentation


class Choice(AbstractAugmentation, audobject.Object):
    def __init__(
        self,
        choices: List[Dict],
        weights: Optional[List[float]] = None,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Choose one augmentation from a list of augmentations with a given
        probability.

        The order of the augmentations in the list is not considered and is
        placed with respect to the order of the choice augmentation itself.

        Augmentations in the list must not have a collate function.

        Args:
            choices: A list of (shorthand syntax) dictionaries defining the
                augmentations to choose from.
            weights: A list of weights for each choice. If None, all
                augmentations are assigned equal weights. Defaults to None.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.

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
        self.augmentation_choices = []

        for aug in self.choices:
            if isinstance(aug, str):
                self.augmentation_choices.append(
                    {aug: {"generator_seed": self.generator_seed}}
                )
            else:
                aug_name = next(iter(aug.keys()))
                if aug[aug_name].get("generator_seed") is None:
                    aug[aug_name]["generator_seed"] = self.generator_seed
                self.augmentation_choices.append(aug)

        self.augmentation_choices: List[AbstractAugmentation] = [
            autrainer.instantiate_shorthand(
                choice,
                instance_of=AbstractAugmentation,
            )
            for choice in self.choices
        ]
        for aug in self.augmentation_choices:
            if hasattr(aug, "get_collate_fn"):
                raise ValueError(
                    "Choice augmentations must not have a collate function."
                )

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        """Choose one augmentation from the list of augmentations based on the
        given weights.

        Args:
            x: The input tensor.
            index: The index of the input tensor in the dataset.
                Defaults to None.

        Returns:
            The augmented tensor.
        """
        weights = torch.Tensor(self.weights)
        choice = torch.multinomial(weights, 1, generator=self.g).item()
        return self.augmentation_choices[choice](x, index)

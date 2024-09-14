from abc import abstractmethod
from typing import Optional

import torch

from autrainer.transforms import AbstractTransform


class AbstractAugmentation(AbstractTransform):
    def __init__(
        self,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Abstract class for an augmentation.

        Args:
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
            kwargs: Additional keyword arguments to store in the object.

        Raises:
            ValueError: If p is not in the range [0, 1].
        """
        super().__init__(order, **kwargs)
        if p < 0 or p > 1:
            raise ValueError("p must be in the range [0, 1]")
        self.p = p
        self.generator_seed = generator_seed
        self.g = torch.Generator()
        if self.generator_seed is not None:
            self.g.manual_seed(self.generator_seed)

    def __call__(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        """Call the augmentation apply method with probability p.

        Args:
            x: The input tensor.
            index: The index of the input tensor in the dataset.
                Defaults to None.

        Returns:
            The augmented tensor if the probability is less than p, otherwise
            the input tensor.
        """
        probability = torch.rand(1, generator=self.g).item()
        return self.apply(x, index) if probability < self.p else x

    @abstractmethod
    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        """Apply the augmentation to the input tensor.

        Apply is called with probability p.

        Args:
            x: The input tensor.
            index: The index of the input tensor in the dataset.
                Defaults to None.

        Returns:
            The augmented tensor.
        """

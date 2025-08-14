from abc import abstractmethod
from typing import Any, Dict, Optional

import torch

from autrainer.core.structs import AbstractDataItem
from autrainer.transforms import AbstractTransform


class AbstractAugmentation(AbstractTransform):
    def __init__(
        self,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        **kwargs: Dict[str, Any],
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

    def offset_generator_seed(self, offset: int) -> None:
        """Offset the generator seed used to draw the probability of applying
        the augmentation. Useful for ensuring reproducibility and
        randomness of augmentations when using multiple workers.

        Args:
            offset: Offset to add to the generator seed. Usually the
                worker index.
        """
        if self.generator_seed is None:
            return
        self.generator_seed += offset
        self.g.manual_seed(self.generator_seed)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        """Call the augmentation apply method with probability p.

        Args:
            item: The input data item.

        Returns:
            The augmented item if the probability is less than p, otherwise
            the input item.
        """
        probability = torch.rand(1, generator=self.g).item()
        return self.apply(item) if probability < self.p else item

    @abstractmethod
    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        """Apply the augmentation to the input tensor.

        Apply is called with probability p.

        Args:
            item: The input data item.

        Returns:
            The augmented item.
        """

from typing import Optional, Tuple

import torch

from autrainer.augmentations import AbstractAugmentation


class AmplitudeScale(AbstractAugmentation):
    def __init__(
        self,
        scale_range: Tuple[float, float],
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Amplitude scaling augmentation. The amplitude is randomly scaled by
        a factor drawn from scale_range.

        Args:
            scale_range: The range of the amplitude scaling factor.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.

        Raises:
            ValueError: If p is not in the range [0, 1].
        """
        super().__init__(order, p, generator_seed)
        self.scale_range = scale_range
        self.scale_g = torch.Generator()
        if self.generator_seed:
            self.scale_g.manual_seed(self.generator_seed)

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        s0, s1 = self.scale_range
        return x * (torch.rand(1, generator=self.scale_g) * (s1 - s0) + s0)

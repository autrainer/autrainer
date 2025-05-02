from typing import TYPE_CHECKING, Callable, List, Optional

import torch
from torchvision.transforms import v2

from autrainer.augmentations import AbstractAugmentation
from autrainer.core.structs import AbstractDataBatch, AbstractDataItem


if TYPE_CHECKING:
    from autrainer.datasets import AbstractDataset


class CutMixUp(AbstractAugmentation):
    def __init__(
        self,
        alpha: float = 1.0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Randomly applies CutMix or MixUp augmentations with a probability
        of 0.5 each.

        Args:
            alpha: Hyperparameter of the Beta distribution. Defaults to 1.0.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        self.alpha = alpha
        self.cut_mix_up_g = torch.Generator()
        if generator_seed:
            self.cut_mix_up_g.manual_seed(generator_seed)

    def get_collate_fn(
        self,
        data: "AbstractDataset",
        default: Callable,
    ) -> Callable:
        self.cutmix = v2.CutMix(num_classes=data.output_dim, alpha=self.alpha)
        self.mixup = v2.MixUp(num_classes=data.output_dim, alpha=self.alpha)

        def _collate_fn(batch: List[AbstractDataItem]) -> AbstractDataBatch:
            probability = torch.rand(1, generator=self.g).item()
            batched: AbstractDataBatch = default(batch)
            if probability < self.p:
                features = batched.features
                target = batched.target
                if probability < 0.5:
                    results = self.cutmix(features, target)
                else:
                    results = self.mixup(features, target)
                batched.features = results[0]
                batched.target = results[1]
                return batched
            batched.target = torch.nn.functional.one_hot(
                batched.target, data.output_dim
            ).float()
            return batched

        return _collate_fn

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        # no-op as the augmentation is applied in the collate function
        return item

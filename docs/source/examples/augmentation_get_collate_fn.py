from typing import TYPE_CHECKING, Callable

import torch.utils.data

from autrainer.augmentations.abstract_augmentation import AbstractAugmentation


if TYPE_CHECKING:
    from autrainer.datasets import AbstractDataset


class ExampleCollateAugmentation(AbstractAugmentation):
    def get_collate_fn(self, data: "AbstractDataset") -> Callable:
        return torch.utils.data.default_collate

from typing import TYPE_CHECKING, Callable

from autrainer.augmentations.abstract_augmentation import AbstractAugmentation


if TYPE_CHECKING:
    from autrainer.datasets import AbstractDataset
    from autrainer.datasets.utils import DataBatch


class ExampleCollateAugmentation(AbstractAugmentation):
    def get_collate_fn(
        self,
        data: "AbstractDataset",
        default: Callable,
    ) -> Callable:
        return DataBatch.collate

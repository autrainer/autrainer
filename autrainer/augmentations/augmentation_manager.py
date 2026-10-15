from omegaconf import DictConfig

import autrainer
from autrainer.transforms import SmartCompose

from .augmentation_pipeline import AugmentationPipeline


class AugmentationManager:
    def __init__(
        self,
        train_augmentation: DictConfig | dict | None = None,
        dev_augmentation: DictConfig | dict | None = None,
        test_augmentation: DictConfig | dict | None = None,
    ) -> None:
        """Manage the creation of the augmentation pipelines for train, dev, and test
        sets.

        Args:
            train_augmentation: Train augmentation configuration.
            dev_augmentation: Dev augmentation configuration.
            test_augmentation: Test augmentation configuration.
        """
        self.train = train_augmentation
        self.dev = dev_augmentation
        self.test = test_augmentation

    def get_augmentations(
        self,
    ) -> tuple[SmartCompose, SmartCompose, SmartCompose]:
        """Get augmentation pipelines for train, dev, and test.

        Returns:
            Tuple of augmentation pipelines for train, dev, and test.
        """
        return (
            self._build_augmentation(self.train),
            self._build_augmentation(self.dev),
            self._build_augmentation(self.test),
        )

    def _build_augmentation(
        self,
        augmentation: DictConfig | dict | None,
    ) -> SmartCompose:
        if augmentation is None:
            return SmartCompose([])
        pipeline = autrainer.instantiate(augmentation, instance_of=AugmentationPipeline)
        return pipeline.create_pipeline()

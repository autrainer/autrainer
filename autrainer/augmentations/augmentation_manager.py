from typing import Dict, Optional, Tuple, Union

from omegaconf import DictConfig

import autrainer
from autrainer.transforms import SmartCompose

from .augmentation_pipeline import AugmentationPipeline


class AugmentationManager:
    def __init__(
        self,
        train_augmentation: Optional[Union[DictConfig, Dict]] = None,
        dev_augmentation: Optional[Union[DictConfig, Dict]] = None,
        test_augmentation: Optional[Union[DictConfig, Dict]] = None,
    ) -> None:
        """Manage the creation of the augmentation pipelines for train, dev,
        and test sets.

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
    ) -> Tuple[SmartCompose, SmartCompose, SmartCompose]:
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
        augmentation: Optional[Union[DictConfig, Dict]],
    ) -> SmartCompose:
        if augmentation is None:
            return SmartCompose([])
        pipeline = autrainer.instantiate(
            augmentation,
            instance_of=AugmentationPipeline,
        )
        return pipeline.create_pipeline()

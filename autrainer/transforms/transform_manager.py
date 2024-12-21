import copy
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

import autrainer

from .abstract_transform import AbstractTransform
from .smart_compose import SmartCompose
from .specific_transforms import GrayscaleToRGB, RGBToGrayscale


class TransformManager:
    def __init__(
        self,
        model_transform: Union[DictConfig, Dict],
        dataset_transform: Union[DictConfig, Dict],
        train_augmentation: Optional[SmartCompose] = None,
        dev_augmentation: Optional[SmartCompose] = None,
        test_augmentation: Optional[SmartCompose] = None,
    ) -> None:
        """Transform manager for composing transforms for the train, dev, and
        test datasets.
        Automatically handles the creation of all transformation pipelines
        including incorporating optional augmentations.

        Args:
            model_transform: The model transform configuration.
            dataset_transform: The dataset transform configuration.
            train_augmentation: The train augmentation pipeline.
                Defaults to None.
            dev_augmentation: The dev augmentation pipeline.
                Defaults to None.
            test_augmentation: The test augmentation pipeline.
                Defaults to None.
        """
        if isinstance(model_transform, DictConfig):
            model_transform = OmegaConf.to_container(model_transform)
        if isinstance(dataset_transform, DictConfig):
            dataset_transform = OmegaConf.to_container(dataset_transform)
        self.model_transform = model_transform
        self.dataset_transform = dataset_transform
        self.train_augmentation = train_augmentation
        self.dev_augmentation = dev_augmentation
        self.test_augmentation = test_augmentation

    def get_transforms(
        self,
    ) -> Tuple[SmartCompose, SmartCompose, SmartCompose]:
        """Get the composed transform pipelines for the train, dev, and test
        datasets.

        Returns:
            The composed transform pipelines for the train, dev, and test
                datasets.
        """
        compatibility = self._match_model_dataset()

        train = self._build("train") + compatibility + self.train_augmentation
        dev = self._build("dev") + compatibility + self.dev_augmentation
        test = self._build("test") + compatibility + self.test_augmentation
        return train, dev, test

    def _build(self, subset: str) -> SmartCompose:
        """Build and combine the model and dataset transforms into a single
        transform. Model transforms override dataset transforms when they share
        the same key (or key + tag).

        Args:
            subset: The subset of the transforms to combine.

        Returns:
            The combined transform.
        """
        combined = []
        seen = set()
        model_transforms = self._combine_subset(self.model_transform, subset)
        dataset_transforms = self._combine_subset(
            self.dataset_transform, subset
        )

        for transform in model_transforms:
            key = self._get_key(transform)
            seen.add(key)
            if isinstance(transform, dict) and transform[key] is None:
                continue
            combined.append(transform)

        combined.extend(
            t for t in dataset_transforms if self._get_key(t) not in seen
        )

        return SmartCompose(
            [
                autrainer.instantiate_shorthand(t, AbstractTransform)
                for t in combined
            ]
        )

    @staticmethod
    def _combine_subset(
        transform: Dict,
        subset: str,
    ) -> List[Union[str, Dict]]:
        """Combine a subset of the transform list.

        Args:
            transform: The model or dataset transform.
            subset: The subset to combine with the base transforms.

        Returns:
            The subset of the transform list.
        """
        base = copy.deepcopy(transform.get("base", []))
        specific = copy.deepcopy(transform.get(subset, []))
        return base + specific

    @staticmethod
    def _get_key(transform: Union[str, Dict]) -> str:
        if isinstance(transform, dict):
            return next(iter(transform.keys()))
        return transform

    def _match_model_dataset(self) -> SmartCompose:
        """Match the model input type with the dataset type by adding
        the necessary transform.

        Raises:
            ValueError: If the model and dataset types do not match.

        Returns:
            The composed transform to match the model and dataset types.
        """
        m = self.model_transform["type"]
        d = self.dataset_transform["type"]
        match = []
        if m == "image" and d == "grayscale":
            match = [GrayscaleToRGB()]
        elif m == "grayscale" and d == "image":
            match = [RGBToGrayscale()]
        elif m != d:
            raise ValueError(
                f"Model input type '{m}' does not match dataset type '{d}'"
            )
        return SmartCompose(match)

from typing import Dict, Optional, Tuple, Union

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
        base = self._build("base") + self._match_model_dataset()

        train = base + self._build("train") + self.train_augmentation
        dev = base + self._build("dev") + self.dev_augmentation
        test = base + self._build("test") + self.test_augmentation
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
        model_transforms = self.model_transform.get(subset, [])
        dataset_transforms = self.dataset_transform.get(subset, [])

        for transform in model_transforms:
            key = self._get_key(transform)
            seen.add(key)

            # allow the model to remove a transform by setting it to None
            # e.g., autrainer.transforms.Normalize: null
            if isinstance(transform, dict) and transform[key] is None:
                continue

            combined.append(transform)

        combined.extend(t for t in dataset_transforms if self._get_key(t) not in seen)

        return SmartCompose([self._instantiate(t) for t in combined])

    @staticmethod
    def _get_key(transform: Union[str, Dict]) -> str:
        if isinstance(transform, dict):
            return next(iter(transform.keys()))
        return transform

    @staticmethod
    def _instantiate(transform: Union[str, Dict]) -> AbstractTransform:
        if isinstance(transform, str):
            t = transform.split("@")[0]
            return autrainer.instantiate_shorthand(t, AbstractTransform)
        key = next(iter(transform.keys()))
        return autrainer.instantiate_shorthand(
            {key.split("@")[0]: transform[key]},
            AbstractTransform,
        )

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

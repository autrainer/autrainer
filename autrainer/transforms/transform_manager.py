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
        base = self._specific("base") + self._match_model_dataset()

        train = base + self._specific("train") + self.train_augmentation
        dev = base + self._specific("dev") + self.dev_augmentation
        test = base + self._specific("test") + self.test_augmentation
        return train, dev, test

    def _specific(self, subset: str) -> SmartCompose:
        """Get and build the specific transforms for the model and dataset
        specified in the transform configurations.

        Args:
            subset: The transform subset to get. Must be one of "train", "dev",
                "test", or "base".

        Raises:
            ValueError: If the subset is not one of "train", "dev", "test",
                or "base".

        Returns:
            The composed transform pipeline.
        """
        if subset not in ["train", "dev", "test", "base"]:
            raise ValueError(
                "Transform subset must be one of 'train', 'dev', 'test', "
                f"or 'base', got {subset}."
            )
        model_transforms = self.model_transform.get(subset, [])
        dataset_transforms = self.dataset_transform.get(subset, [])
        if not model_transforms and not dataset_transforms:
            return SmartCompose([])
        model_transforms, dataset_transforms = self._filter_transforms(
            model_transforms, dataset_transforms
        )
        combined_transforms = self._combine_transforms(
            model_transforms,
            dataset_transforms,
        )
        transforms = [
            autrainer.instantiate_shorthand(t, instance_of=AbstractTransform)
            for t in combined_transforms
        ]
        return SmartCompose(transforms)

    def _filter_transforms(
        self,
        model_transforms: List[Union[str, Dict]],
        dataset_transforms: List[Union[str, Dict]],
    ) -> Tuple[List[Union[str, Dict]], List[Union[str, Dict]]]:
        """Remove None values from the model and dataset transforms.
        None values are used to indicate that the transform was removed
        from the config by either the model or dataset transform.
        See `Automatic Transforms` for more details.

        Args:
            model_transforms: List of model transforms.
            dataset_transforms: List of dataset transforms.

        Returns:
            The filtered model and dataset transforms.
        """

        def _filter(l1, l2):
            for transform in l1.copy():
                if isinstance(transform, str):
                    continue
                key = next(iter(transform.keys()))
                if transform[key] is None and len(transform.keys()) == 1:
                    l1.remove(transform)
                    d_idx = self._find_matching_transform(l2, key)
                    if d_idx is not None:
                        l2.pop(d_idx)
            return l1, l2

        model_transforms, dataset_transforms = _filter(
            model_transforms, dataset_transforms
        )
        dataset_transforms, model_transforms = _filter(
            dataset_transforms, model_transforms
        )
        return model_transforms, dataset_transforms

    def _combine_transforms(
        self,
        model_transforms: List[Union[str, Dict]],
        dataset_transforms: List[Union[str, Dict]],
    ) -> List[Union[str, Dict]]:
        """Combine model and dataset transforms into a single list.
        Model transforms outweigh dataset transforms, so if a transform
        is present in both lists, the model transform will be used.

        Args:
            model_transforms: List of model transforms.
            dataset_transforms: List of dataset transforms.

        Returns:
            The combined list of transforms.
        """
        combined = model_transforms
        for transform in dataset_transforms:
            if isinstance(transform, str):
                if transform not in combined:
                    combined.append(transform)
            else:
                key = next(iter(transform.keys()))
                if self._find_matching_transform(combined, key) is None:
                    combined.append(transform)
        return combined

    @staticmethod
    def _find_matching_transform(
        l: List[Union[str, Dict]],
        key: str,
    ) -> Optional[int]:
        """Find the index of a transform in a list of transforms.
        If the transform is a dictionary, the key is used to find the
        transform. If the transform is a string, the string itself is
        used to find the transform.

        Args:
            l: List of transforms.
            key: The key to search for.

        Returns:
            The index of the transform in the list. If the transform is not
                found, None is returned.
        """
        x = (
            idx
            for idx, t in enumerate(l)
            if (isinstance(t, dict) and key in t.keys()) or t == key
        )
        return next(x, None)

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

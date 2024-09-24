from typing import Optional

import torch

import autrainer

from .abstract_augmentation import AbstractAugmentation


try:
    import albumentations  # noqa: F401

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover
    ALBUMENTATIONS_AVAILABLE = False  # pragma: no cover

try:
    import audiomentations  # noqa: F401

    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover
    AUDIOMENTATIONS_AVAILABLE = False  # pragma: no cover

try:
    import torch_audiomentations  # noqa: F401

    TORCHAUDIOMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCHAUDIOMENTATIONS_AVAILABLE = False  # pragma: no cover


class AugmentationWrapper(AbstractAugmentation):
    def __init__(
        self,
        augmentation_import_path: AbstractAugmentation,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        pass_index: bool = False,
        probability_attr: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Wrapper around an AbstractAugmentation.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            augmentation_import_path: The augmentation import path for the
                underlying augmentation to wrap.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
            pass_index: Whether to pass the index to the underlying
                augmentation. Defaults to False.
            probability_attr: Name of the underlying augmentation's
                probability attribute to override with 1.0. If None, the
                probability attribute is not overridden. Defaults to None.
            kwargs: Additional keyword arguments to store in the object.
        """
        super().__init__(order, p, generator_seed, **kwargs)
        self.augmentation_import_path = augmentation_import_path
        self._deterministic = False

        # ? audobject passes _object_root_ to the object, which is not a valid
        # ? argument for the base augmentation class.
        kwargs.pop("_object_root_", None)
        self.augmentation = autrainer.instantiate_shorthand(
            self.augmentation_import_path,
            **kwargs,
        )

        self.pass_index = pass_index
        self.probability_attr = probability_attr
        if self.probability_attr and hasattr(
            self.augmentation, self.probability_attr
        ):
            setattr(self.augmentation, self.probability_attr, 1.0)

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        if self.pass_index:
            return self.augmentation(x, index)  # pragma: no cover
        return self.augmentation(x)


class TorchvisionAugmentation(AugmentationWrapper):
    def __init__(
        self,
        name: str,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Wrapper around torchvision.transforms.v2 transforms, which are
        specified by their class name and keyword arguments.

        Functionals are currently not supported.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            name: Name of the torchvision augmentation. Must be a valid
                torchvision.transforms.v2 transform class name.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
            kwargs: Keyword arguments passed to the torchvision augmentation.
        """
        self.name = name

        super().__init__(
            augmentation_import_path=f"torchvision.transforms.v2.{self.name}",
            order=order,
            p=p,
            generator_seed=generator_seed,
            pass_index=False,
            probability_attr="p",
            **kwargs,
        )


class TorchaudioAugmentation(AugmentationWrapper):
    def __init__(
        self,
        name: str,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Wrapper around torchaudio.transforms transforms, which are specified
        by their class name and keyword arguments.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            name: Name of the torchaudio augmentation. Must be a valid
                torchaudio.transforms transform class name.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
            kwargs: Keyword arguments passed to the torchaudio augmentation.
        """
        self.name = name
        super().__init__(
            augmentation_import_path=f"torchaudio.transforms.{self.name}",
            order=order,
            p=p,
            generator_seed=generator_seed,
            pass_index=False,
            probability_attr="p",
            **kwargs,
        )

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        x = super().apply(x, index)
        if isinstance(x, tuple):
            return x[0]
        return x


class AlbumentationsAugmentation(AugmentationWrapper):
    def __init__(
        self,
        name: str,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Wrapper around albumentations transforms, which are specified by
        their class name and keyword arguments.

        Albumentations operates on numpy arrays, so the input tensor is
        converted to a numpy array before applying the augmentation, and the
        output numpy array is converted back to a tensor.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            name: Name of the albumentations augmentation. Must be a valid
                albumentations transform class name.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
            kwargs: Keyword arguments passed to the albumentations
                augmentation.
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError(
                "Albumentations is not installed. Install the required extras "
                "with 'pip install autrainer[albumentations]'."
            )  # pragma: no cover

        self.name = name
        super().__init__(
            augmentation_import_path=f"albumentations.{self.name}",
            order=order,
            p=p,
            generator_seed=generator_seed,
            pass_index=False,
            probability_attr="p",
            **kwargs,
        )

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        device = x.device
        x = x.cpu().permute(1, 2, 0).numpy()
        x = self.augmentation(image=x)["image"]
        x = torch.from_numpy(x).permute(2, 0, 1).to(device)
        return x


class AudiomentationsAugmentation(AugmentationWrapper):
    def __init__(
        self,
        name: str,
        sample_rate: Optional[int] = None,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Wrapper around audiomentations transforms, which are specified
        by their class name and keyword arguments.

        Audiomentations operates on numpy arrays, so the input tensor is
        converted to a numpy array before applying the augmentation, and the
        output numpy array is converted back to a tensor.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            name: Name of the torchaudio augmentation. Must be a valid
                audiomentations transform class name.
            sample_rate: The sample rate of the audio data. Should be specified
                for most audio augmentations. If None, the sample rate is not
                passed to the augmentation. Defaults to None.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
            kwargs: Keyword arguments passed to the audiomentations
                augmentation.
        """
        if not AUDIOMENTATIONS_AVAILABLE:
            raise ImportError(
                "Audiomentations is not installed. Install the required "
                "extras with 'pip install autrainer[audiomentations]'."
            )  # pragma: no cover

        self.name = name
        self.sample_rate = sample_rate
        self._aug_kwargs = {}
        if self.sample_rate is not None:
            self._aug_kwargs["sample_rate"] = self.sample_rate
        super().__init__(
            augmentation_import_path=f"audiomentations.{self.name}",
            order=order,
            p=p,
            generator_seed=generator_seed,
            pass_index=False,
            probability_attr="p",
            **kwargs,
        )

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        device = x.device
        x = self.augmentation(x.cpu().numpy(), **self._aug_kwargs)
        return torch.from_numpy(x).to(device)


class TorchAudiomentationsAugmentation(AugmentationWrapper):
    def __init__(
        self,
        name: str,
        sample_rate: int,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Wrapper around torch_audiomentations transforms, which are specified
        by their class name and keyword arguments.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            name: Name of the torchaudio augmentation. Must be a valid
                torch_audiomentations transform class name.
            sample_rate: The sample rate of the audio data.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
            kwargs: Keyword arguments passed to the torch_audiomentations
                augmentation.
        """
        if not TORCHAUDIOMENTATIONS_AVAILABLE:
            raise ImportError(
                "torch-audiomentations is not installed. Install the required "
                "extras with 'pip install autrainer[torch-audiomentations]'."
            )  # pragma: no cover

        self.name = name
        self.sample_rate = sample_rate
        kwargs["output_type"] = "dict"
        super().__init__(
            augmentation_import_path=f"torch_audiomentations.{self.name}",
            order=order,
            p=p,
            generator_seed=generator_seed,
            pass_index=False,
            probability_attr="p",
            **kwargs,
        )

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        x = x.unsqueeze(0)
        x = self.augmentation(x, sample_rate=self.sample_rate).samples
        return x.squeeze(0)

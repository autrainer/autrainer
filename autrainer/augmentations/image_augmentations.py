import os
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type

import pandas as pd
import torch
from torch.utils.data import default_collate
from torchvision.transforms import v2

from .abstract_augmentation import AbstractAugmentation


if TYPE_CHECKING:
    from autrainer.datasets import AbstractDataset  # pragma: no cover


class BaseMixUpCutMix(AbstractAugmentation):
    def __init__(
        self,
        augmentation_class: Type,
        alpha: float = 1.0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Base class for MixUp and CutMix augmentations.

        Args:
            augmentation_class: The class of the augmentation to apply.
            alpha: Hyperparameter of the Beta distribution. Defaults to 1.0.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        self.augmentation_class = augmentation_class
        self.alpha = alpha

    def get_collate_fn(self, data: "AbstractDataset") -> Callable:
        if data.output_dim <= 1:
            raise ValueError(
                f"{self.augmentation_class.__name__} "
                "requires more than 1 class."
            )
        self.augmentation = self.augmentation_class(
            num_classes=data.output_dim, alpha=self.alpha
        )

        def _collate_fn(
            batch: List[Tuple[torch.Tensor, int, int]],
        ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            probability = torch.rand(1, generator=self.g).item()
            if probability < self.p:
                return self.augmentation(*default_collate(batch))

            batched = default_collate(batch)
            batched[1] = torch.nn.functional.one_hot(
                batched[1], data.output_dim
            ).float()
            return batched

        return _collate_fn

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        return x


class MixUp(BaseMixUpCutMix):
    def __init__(
        self,
        alpha: float = 1.0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """MixUp augmentation. As MixUp utilizes a collate function, the
        probability of applying the augmentation is drawn for each batch.

        Args:
            alpha: Hyperparameter of the Beta distribution. Defaults to 1.0.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(v2.MixUp, alpha, order, p, generator_seed)


class CutMix(BaseMixUpCutMix):
    def __init__(
        self,
        alpha: float = 1.0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """CutMix augmentation. As CutMix utilizes a collate function, the
        probability of applying the augmentation is drawn for each batch.

        Args:
            alpha: Hyperparameter of the Beta distribution. Defaults to 1.0.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(v2.CutMix, alpha, order, p, generator_seed)


class SampleGaussianWhiteNoise(AbstractAugmentation):
    def __init__(
        self,
        snr_df: str,
        snr_col: str,
        sample_seed: int = None,
        order: int = 101,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Sample-level gaussian white noise augmentation based on SNR values.

        Args:
            snr_df: Path to a CSV file containing SNR values for each sample.
                Index of the CSV file must match the index of the dataset.
            snr_col: Name of the column containing the SNR values.
            sample_seed: Seed for the random number generator used for sampling
                the noise. If a seed is provided, a consistent augmentation
                is applied to the same sample. Defaults to None.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 101.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        if not os.path.exists(snr_df):
            raise FileNotFoundError(f"File {snr_df} does not exist.")
        self.snr_df = snr_df
        self.snr_col = snr_col
        self.sample_seed = sample_seed
        df = pd.read_csv(self.snr_df)
        if self.snr_col not in df.columns:
            raise ValueError(
                f"Column {self.snr_col} not found in {self.snr_df}."
            )
        self.snr = df[self.snr_col]
        self._generator = torch.Generator()

    def apply(self, x: torch.Tensor, index: int = None) -> torch.Tensor:
        if self.sample_seed:
            self._generator.manual_seed(
                hash((self.sample_seed, index)) & 0xFFFFFFFF
            )

        snr = 10 ** (self.snr[index] / 10)
        energy = torch.mean(x**2)
        noise = torch.normal(0, 1, generator=self._generator, size=x.shape)
        noise_energy = torch.mean(noise**2)
        scale = torch.sqrt(energy / (snr * noise_energy))
        return x + noise * scale

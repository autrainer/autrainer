import os
import tempfile
from typing import Any, Callable, Dict, Tuple, Type, Union

from omegaconf import DictConfig, ListConfig
import pandas as pd
import pytest
import torch

from autrainer.augmentations import (
    AbstractAugmentation,
    AlbumentationsAugmentation,
    AudiomentationsAugmentation,
    AugmentationManager,
    Choice,
    CutMix,
    FrequencyMask,
    GaussianNoise,
    MixUp,
    SampleGaussianWhiteNoise,
    Sequential,
    SpecAugment,
    TimeMask,
    TimeShift,
    TimeWarp,
    TorchaudioAugmentation,
    TorchAudiomentationsAugmentation,
    TorchvisionAugmentation,
)
from autrainer.augmentations.image_augmentations import BaseMixUpCutMix
from autrainer.transforms import SmartCompose


AUGMENTATION_SEED = 42
AUGMENTATION_FIXTURES = [
    (
        AlbumentationsAugmentation,
        {"name": "Posterize"},
        (3, 32, 32),
        (3, 32, 32),
    ),
    (
        AudiomentationsAugmentation,
        {"name": "PitchShift", "sample_rate": 16000},
        (1, 16000),
        (1, 16000),
    ),
    (
        Choice,
        {"choices": [{"autrainer.augmentations.GaussianNoise": {}}]},
        (3, 32, 32),
        (3, 32, 32),
    ),
    (CutMix, {}, (3, 32, 32), (3, 32, 32)),
    (FrequencyMask, {"freq_mask": 3, "axis": 1}, (3, 32, 32), (3, 32, 32)),
    (GaussianNoise, {}, (3, 32, 32), (3, 32, 32)),
    (MixUp, {}, (3, 32, 32), (3, 32, 32)),
    (
        Sequential,
        {"sequence": [{"autrainer.augmentations.GaussianNoise": {}}]},
        (3, 32, 32),
        (3, 32, 32),
    ),
    (
        SpecAugment,
        {"time_mask": 3, "freq_mask": 3, "W": 3},
        (3, 32, 32),
        (3, 32, 32),
    ),
    (TimeMask, {"time_mask": 3, "axis": 0}, (3, 32, 32), (3, 32, 32)),
    (TimeShift, {"time_steps": 3, "axis": 0}, (3, 32, 32), (3, 32, 32)),
    (TimeWarp, {"W": 3, "axis": 0}, (3, 32, 32), (3, 32, 32)),
    (
        TorchaudioAugmentation,
        {"name": "Speed", "orig_freq": 16000, "factor": 2.0},
        (1, 16000),
        (1, 8000),
    ),
    (
        TorchaudioAugmentation,
        {"name": "FrequencyMasking", "freq_mask_param": 10},
        (1, 101, 64),
        (1, 101, 64),
    ),
    (
        TorchAudiomentationsAugmentation,
        {"name": "AddColoredNoise", "sample_rate": 16000, "min_f_decay": -1},
        (1, 16000),
        (1, 16000),
    ),
    (
        TorchvisionAugmentation,
        {"name": "RandomGrayscale"},
        (3, 32, 32),
        (3, 32, 32),
    ),
]


class TestAllAugmentations:
    @pytest.mark.parametrize(
        "augmentation, params, input_shape, output_shape",
        AUGMENTATION_FIXTURES,
    )
    def test_probability(
        self,
        augmentation: Type[AbstractAugmentation],
        params: Dict[str, Any],
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        output_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> None:
        with pytest.raises(ValueError):
            augmentation(**params, generator_seed=AUGMENTATION_SEED, p=1.1)
        with pytest.raises(ValueError):
            augmentation(**params, generator_seed=AUGMENTATION_SEED, p=-0.1)
        instance = augmentation(
            **params,
            generator_seed=AUGMENTATION_SEED,
            p=0,
        )
        if len(input_shape) == 2:
            x = torch.randn(*input_shape)
        else:
            x = torch.randint(0, 255, input_shape, dtype=torch.uint8)
        assert torch.allclose(x, instance(x)), "Should not apply augmentation"

    @pytest.mark.parametrize(
        "augmentation, params, input_shape, output_shape",
        AUGMENTATION_FIXTURES,
    )
    def test_deterministic(
        self,
        augmentation: Type[AbstractAugmentation],
        params: Dict[str, Any],
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        output_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> None:
        if len(input_shape) == 2:
            x = torch.randn(*input_shape)
        else:
            x = torch.randint(0, 255, input_shape, dtype=torch.uint8)
        aug1 = augmentation(**params, generator_seed=AUGMENTATION_SEED)
        aug2 = augmentation(**params, generator_seed=AUGMENTATION_SEED)
        if hasattr(aug1, "_deterministic") and not aug1._deterministic:
            return  # Skip if known to be non-deterministic
        y1 = aug1(x)
        y2 = aug2(x)
        assert torch.allclose(y1, y2), "Should be deterministic"
        y3 = aug1(x)
        y4 = aug2(x)
        assert torch.allclose(y3, y4), "Should be deterministic"


class TestAugmentationManagerPipeline:
    pipline_cfg1 = {
        "id": "TestGaussianNoise1",
        "_target_": "autrainer.augmentations.AugmentationPipeline",
        "pipeline": ListConfig(["autrainer.augmentations.GaussianNoise"]),
    }
    pipline_cfg2 = {
        "id": "TestGaussianNoise2",
        "_target_": "autrainer.augmentations.AugmentationPipeline",
        "pipeline": [{"autrainer.augmentations.GaussianNoise": {"mean": 0.0}}],
    }

    @pytest.mark.parametrize(
        "config, subset",
        [
            (None, "train"),
            (None, "dev"),
            (None, "test"),
            (pipline_cfg1, "train"),
            (pipline_cfg2, "dev"),
            (pipline_cfg1, "test"),
            (DictConfig(pipline_cfg2), "train"),
            (DictConfig(pipline_cfg1), "dev"),
            (DictConfig(pipline_cfg2), "test"),
        ],
    )
    def test_creation(
        self, config: Union[DictConfig, Dict, None], subset: str
    ) -> None:
        am = AugmentationManager(**{f"{subset}_augmentation": config})
        augs = {}
        augs["train"], augs["dev"], augs["test"] = am.get_augmentations()
        assert all(
            isinstance(aug, SmartCompose) for aug in augs.values()
        ), "Should return SmartCompose instances"
        if config is None:
            assert all(
                not a.transforms for a in augs.values()
            ), "All augmentations should be empty"
        else:
            assert (
                len(augs.pop(subset).transforms) == 1
            ), "Should have 1 augmentation"
            assert all(
                not a.transforms for a in augs.values()
            ), "All other augmentations should be empty"


class TestChoice:
    def test_invalid_weights(self) -> None:
        with pytest.raises(ValueError):
            Choice(
                choices=["autrainer.augmentations.GaussianNoise"],
                weights=[0.5, 0.5],
            )

    def test_invalid_collate_fn(self) -> None:
        with pytest.raises(ValueError):
            Choice(choices=["autrainer.augmentations.CutMix"])


class TestSequential:
    def test_invalid_collate_fn(self) -> None:
        with pytest.raises(ValueError):
            Sequential(sequence=["autrainer.augmentations.CutMix"])


class TestBaseMixUpCutMix:
    @classmethod
    def setup_class(cls) -> None:
        class MockClassificationDataset:
            output_dim = 10

        class MockRegressionDataset:
            output_dim = 1

        cls.classification_dataset = MockClassificationDataset()
        cls.regression_dataset = MockRegressionDataset()

    @pytest.mark.parametrize("aug", [MixUp, CutMix])
    def test_invalid_dataset(self, aug: Type[BaseMixUpCutMix]) -> None:
        with pytest.raises(ValueError):
            aug().get_collate_fn(self.regression_dataset)

    @pytest.mark.parametrize("aug", [MixUp, CutMix])
    def test_collate_fn(self, aug: Type[BaseMixUpCutMix]) -> None:
        self._test_collate(aug().get_collate_fn(self.classification_dataset))

    @pytest.mark.parametrize("aug", [MixUp, CutMix])
    def test_collate_identity(self, aug: Type[BaseMixUpCutMix]) -> None:
        (x_in, y_in, idx_in), (x_out, y_out, idx_out) = self._test_collate(
            aug(p=0).get_collate_fn(self.classification_dataset)
        )
        assert torch.allclose(x_in, x_out), "Should be the same"
        assert torch.allclose(y_in, y_out), "Should be the same"
        assert torch.allclose(idx_in, idx_out), "Should be the same"

    def _test_collate(self, collate_fn: Callable) -> Tuple[Tuple, Tuple]:
        x1, x2 = torch.randn(3, 32, 32), torch.randn(3, 32, 32)
        y1, y2 = 0, 1
        idx1, idx2 = 0, 1
        x_out, y_out, idx_out = collate_fn([(x1, y1, idx1), (x2, y2, idx2)])
        x_in = torch.cat([x1.unsqueeze(0), x2.unsqueeze(0)], dim=0)
        y_in = torch.nn.functional.one_hot(
            torch.tensor([y1, y2]), self.classification_dataset.output_dim
        ).float()
        idx_in = torch.tensor([idx1, idx2])
        assert x_in.shape == x_out.shape, "Should have same x shape"
        assert y_in.shape == y_out.shape, "Should have same y shape"
        assert idx_in.shape == idx_out.shape, "Should have same idx shape"
        return (x_in, y_in, idx_in), (x_out, y_out, idx_out)


class TestSampleGaussianWhiteNoise:
    @classmethod
    def setup_class(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_csv_path = os.path.join(cls.temp_dir.name, "test.csv")
        data = {"snr": [10, 20, 30]}
        pd.DataFrame(data).to_csv(cls.temp_csv_path, index=False)

    @classmethod
    def teardown_class(cls) -> None:
        cls.temp_dir.cleanup()

    def test_invalid_csv(self) -> None:
        with pytest.raises(FileNotFoundError):
            SampleGaussianWhiteNoise(
                snr_df=os.path.join(self.temp_dir.name, "invalid.csv"),
                snr_col="snr",
            )
        with pytest.raises(ValueError):
            SampleGaussianWhiteNoise(
                snr_df=self.temp_csv_path,
                snr_col="invalid",
            )

    def test_sample_noise(self) -> None:
        aug = SampleGaussianWhiteNoise(
            snr_df=self.temp_csv_path,
            snr_col="snr",
            sample_seed=AUGMENTATION_SEED,
            generator_seed=AUGMENTATION_SEED,
        )
        g = torch.Generator()
        x1 = torch.randn(1, 16000)
        x2 = torch.randn(1, 16000)
        x3 = torch.randn(1, 16000)
        c1 = self._mock_snr_calculation(x1, 0, 10, g)
        c2 = self._mock_snr_calculation(x2, 1, 20, g)
        c3 = self._mock_snr_calculation(x3, 2, 30, g)
        y1, y2, y3 = aug(x1, 0), aug(x2, 1), aug(x3, 2)
        assert torch.allclose(y1, c1), "Should apply noise"
        assert torch.allclose(y2, c2), "Should apply noise"
        assert torch.allclose(y3, c3), "Should apply noise"

        c4 = self._mock_snr_calculation(x1, 0, 10, g)
        y4 = aug(x1, 0)
        assert torch.allclose(y4, c4), "Should be deterministic"

    def _mock_snr_calculation(
        self,
        x: torch.Tensor,
        index: int,
        snr: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        generator.manual_seed(hash((AUGMENTATION_SEED, index)) & 0xFFFFFFFF)
        snr = 10 ** (snr / 10)
        energy = torch.mean(x**2)
        noise = torch.normal(0, 1, generator=generator, size=x.shape)
        noise_energy = torch.mean(noise**2)
        scale = torch.sqrt(energy / (snr * noise_energy))
        return x + noise * scale


class TestTimeShift:
    def test_invalid_time_steps(self) -> None:
        with pytest.raises(ValueError):
            TimeShift(time_steps=-1, axis=0)

    def test_identity(self) -> None:
        x = torch.randn(1, 101, 64)
        y = TimeShift(time_steps=0, axis=0)(x)
        assert torch.allclose(x, y), "Should be the same"

    @pytest.mark.parametrize("axis", [0, 1])
    def test_time_shift(self, axis: int) -> None:
        x = torch.randn(1, 101, 64)
        y = TimeShift(time_steps=3, axis=axis)(x)
        assert x.shape == y.shape, "Should have same shape"


class TestTimeMask:
    def test_invalid_time_mask(self) -> None:
        with pytest.raises(ValueError):
            TimeMask(time_mask=-1, axis=0)

    def test_identity(self) -> None:
        x = torch.randn(1, 101, 64)
        y = TimeMask(time_mask=0, axis=0)(x)
        assert torch.allclose(x, y), "Should be the same"

    @pytest.mark.parametrize("axis", [0, 1])
    def test_time_mask(self, axis: int) -> None:
        x = torch.randn(1, 101, 64)
        y = TimeMask(time_mask=3, axis=axis)(x)
        assert x.shape == y.shape, "Should have same shape"


class TestFrequencyMask:
    def test_invalid_freq_mask(self) -> None:
        with pytest.raises(ValueError):
            FrequencyMask(freq_mask=-1, axis=0)

    def test_identity(self) -> None:
        x = torch.randn(1, 101, 64)
        y = FrequencyMask(freq_mask=0, axis=0)(x)
        assert torch.allclose(x, y), "Should be the same"

    @pytest.mark.parametrize("axis", [0, 1])
    def test_freq_mask(self, axis: int) -> None:
        x = torch.randn(1, 101, 64)
        y = FrequencyMask(freq_mask=3, axis=axis)(x)
        assert x.shape == y.shape, "Should have same shape"


class TestTimeWarp:
    @pytest.mark.parametrize("axis", [0, 1])
    def test_time_warp(self, axis: int) -> None:
        x = torch.randn(1, 101, 64)
        y = TimeWarp(W=3, axis=axis)(x)
        assert x.shape == y.shape, "Should have same shape"

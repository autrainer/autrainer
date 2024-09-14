from typing import Any, Callable, Dict, List, Tuple, Type, Union
import warnings

import numpy as np
from omegaconf import DictConfig
from PIL import Image
import pytest
import torch
from torchvision import transforms as T

from autrainer.augmentations import AbstractAugmentation, CutMix
from autrainer.transforms import (
    AbstractTransform,
    AnyToTensor,
    Expand,
    FeatureExtractor,
    GrayscaleToRGB,
    Normalize,
    NumpyToTensor,
    OpenSMILE,
    PannMel,
    RandomCrop,
    Resample,
    Resize,
    RGBAToGrayscale,
    RGBAToRGB,
    RGBToGrayscale,
    ScaleRange,
    SmartCompose,
    SpectToImage,
    SquarePadCrop,
    StereoToMono,
    TransformManager,
)

from .test_augmentations import AUGMENTATION_FIXTURES, AUGMENTATION_SEED


# TODO: Correctness test for the transforms

TRANSFORM_FIXTURES = [
    (AnyToTensor, {}, (3, 32, 32), (3, 32, 32)),
    (Expand, {"size": 16000, "axis": -1}, (1, 8000), (1, 16000)),
    (GrayscaleToRGB, {}, (1, 32, 32), (3, 32, 32)),
    (NumpyToTensor, {}, (3, 32, 32), (3, 32, 32)),
    (Normalize, {"mean": [0.0], "std": [1.0]}, (3, 32, 32), (3, 32, 32)),
    (RandomCrop, {"size": 8000, "axis": -1}, (1, 16000), (1, 8000)),
    (
        Resample,
        {"current_sr": 16000, "target_sr": 44100},
        (1, 16000),
        (1, 44100),
    ),
    (Resize, {"height": 64, "width": 128}, (3, 32, 32), (3, 64, 128)),
    (RGBAToGrayscale, {}, (4, 32, 32), (1, 32, 32)),
    (RGBAToRGB, {}, (4, 32, 32), (3, 32, 32)),
    (RGBToGrayscale, {}, (3, 32, 32), (1, 32, 32)),
    (ScaleRange, {}, (3, 32, 32), (3, 32, 32)),
    (SpectToImage, {"height": 32, "width": 32}, (1, 32, 128), (3, 32, 32)),
    (SquarePadCrop, {"mode": "pad"}, (3, 32, 64), (3, 64, 64)),
    (SquarePadCrop, {"mode": "crop"}, (3, 32, 64), (3, 32, 32)),
    (StereoToMono, {}, (2, 16000), (1, 16000)),
]


class TestAllTransforms:
    @pytest.mark.parametrize(
        "transform, params, input_shape, output_shape",
        TRANSFORM_FIXTURES + AUGMENTATION_FIXTURES,
    )
    def test_output_shape(
        self,
        transform: Type[Union[AbstractTransform, AbstractAugmentation]],
        params: Dict[str, Any],
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        output_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> None:
        if issubclass(transform, AbstractAugmentation):
            instance = transform(**params, generator_seed=AUGMENTATION_SEED)
        else:
            instance = transform(**params)
        if len(input_shape) == 2:
            x = torch.randn(*input_shape, dtype=torch.float32)
        else:
            x = torch.randint(0, 255, input_shape, dtype=torch.uint8)
        y = instance(x)
        assert (
            y.shape == output_shape
        ), "Output shape should match the expected shape"
        assert (
            y.device == x.device
        ), "Output device should match the input device"

    def test_repr(self) -> None:
        assert (
            repr(AnyToTensor()) == "AnyToTensor(order=-100)"
        ), "Representation should match the expected string"


class TestAnyToTensor:
    def test_tensor(self) -> None:
        x = torch.randn(3, 32, 32)
        y = AnyToTensor()(x)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.dtype == torch.float32, "Output should be float32"
        assert (
            y.device == x.device
        ), "Output device should match the input device"

    def test_numpy(self) -> None:
        x = np.random.rand(3, 32, 32)
        x = x.astype(np.float32)
        y = AnyToTensor()(x)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.shape == torch.Size(
            [3, 32, 32]
        ), "Output shape should match input"
        assert y.dtype == torch.float32, "Output should be float32"
        assert y.device == torch.device("cpu"), "Output should be on CPU"

    def test_pil(self) -> None:
        x = Image.fromarray(np.random.rand(32, 32, 3).astype(np.uint8))
        y = AnyToTensor()(x)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.shape == torch.Size(
            [3, 32, 32]
        ), "Output shape should match input"
        assert y.dtype == torch.uint8, "Output should be uint8"
        assert y.device == torch.device("cpu"), "Output should be on CPU"

    def test_wrong_input(self) -> None:
        with pytest.raises(TypeError):
            AnyToTensor()(1)


class TestNumpyToTensor:
    def test_numpy(self) -> None:
        x = np.random.rand(3, 32, 32)
        x = x.astype(np.float32)
        y = NumpyToTensor()(x)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.shape == torch.Size(
            [3, 32, 32]
        ), "Output shape should match input"
        assert y.dtype == torch.float32, "Output should be float32"
        assert y.device == torch.device("cpu"), "Output should be on CPU"


class TestPannMel:
    def test_spectrogram(self) -> None:
        x = torch.randn(1, 16000 * 3)
        transform = PannMel(
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            ref=1.0,
            amin=1e-10,
            top_db=None,
        )
        y = transform(x)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.shape == torch.Size(
            [1, 301, 64]
        ), "Output shape should match input"
        assert y.dtype == torch.float32, "Output should be float32"
        assert y.device == torch.device("cpu"), "Output should be on CPU"


class TestResize:
    @pytest.mark.parametrize(
        "height, width, expected",
        [
            (64, "Any", (3, 64, 64)),
            ("Any", 128, (3, 128, 128)),
            ("Any", "Any", None),
        ],
    )
    def test_resize(
        self,
        height: Union[int, str],
        width: Union[int, str],
        expected: Union[Tuple[int, int, int], None],
    ) -> None:
        x = torch.randn(3, 32, 32)
        if not expected:
            with pytest.raises(ValueError):
                Resize(height=height, width=width)
        else:
            assert (
                Resize(height=height, width=width)(x).shape == expected
            ), "Output shape should match the expected shape"


class TestSquarePadCrop:
    @pytest.mark.parametrize(
        "mode, input_shape, expected",
        [
            ("pad", (3, 32, 64), (3, 64, 64)),
            ("crop", (3, 128, 64), (3, 64, 64)),
            ("fizz", (3, 128, 64), None),
        ],
    )
    def test_crop(
        self,
        mode: str,
        input_shape: Tuple[int, int, int],
        expected: Union[Tuple[int, int, int], None],
    ) -> None:
        x = torch.randn(*input_shape)
        if not expected:
            with pytest.raises(ValueError):
                SquarePadCrop(mode=mode)
        else:
            assert (
                SquarePadCrop(mode=mode)(x).shape == expected
            ), "Output shape should match the expected shape"


class TestScaleRange:
    @pytest.mark.parametrize("range", [(0, 1), (-1, 1), (0, 255)])
    def test_range_uint8(self, range: Tuple[int, int]) -> None:
        self._test_range(
            torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8),
            range,
        )

    @pytest.mark.parametrize("range", [(0, 1), (-1, 1), (0, 255)])
    def test_range_float32(self, range: Tuple[int, int]) -> None:
        self._test_range(torch.randn(3, 32, 32), range)

    def test_range_zeros(self) -> None:
        x = torch.zeros(3, 32, 32)
        y = ScaleRange((0, 1))(x)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.shape == torch.Size(
            [3, 32, 32]
        ), "Output shape should match input"
        assert y.dtype == torch.float32, "Output should be float32"
        assert y.min() == 0, "Min value should be 0"
        assert y.max() == 0, "Max value should be 0"

    @pytest.mark.parametrize("range", [(0, 0), (1, 2, 3)])
    def test_range_invalid(self, range: Tuple[int, int]) -> None:
        with pytest.raises(ValueError):
            ScaleRange(range)

    def _test_range(self, x: torch.Tensor, range: Tuple[int, int]) -> None:
        y = ScaleRange(range)(x)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.shape == torch.Size(
            [3, 32, 32]
        ), "Output shape should match input"
        assert y.dtype == torch.float32, "Output should be float32"
        assert y.min() == range[0], "Min value should match the range"
        assert y.max() == range[1], "Max value should match the range"


class TestNormalize:
    @pytest.mark.parametrize(
        "data",
        [
            torch.randn(3, 32, 32),
            torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8),
        ],
    )
    def test_normalize(self, data: torch.Tensor) -> None:
        y = Normalize(mean=[0.0], std=[1.0])(data)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.shape == data.shape, "Output shape should match input"
        assert y.dtype == torch.float32, "Output should be float32"


class TestFeatureExtractor:
    @pytest.mark.parametrize(
        "fe_type, fe_transfer, input_shape, expected",
        [
            (
                "AST",
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                (1, 16000),
                (1024, 128),
            ),
            (
                None,
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                (1, 16000),
                (1024, 128),
            ),
            ("AST", None, (1, 16000), (1024, 128)),
            ("Whisper", None, (1, 16000), (80, 3000)),
            ("W2V2", None, (1, 16000), (16000,)),
            ("AST", None, (2, 16000), (1024, 128)),
        ],
    )
    def test_extractor(
        self,
        fe_type: Union[str, None],
        fe_transfer: Union[str, None],
        input_shape: Tuple[int, int],
        expected: Union[Tuple[int], Tuple[int, int]],
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fe = FeatureExtractor(fe_type, fe_transfer)
        x = torch.randn(*input_shape)
        y = fe(x)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.shape == torch.Size(
            expected
        ), "Output shape should match the expected shape"

    @pytest.mark.parametrize("fe_type, fe_transfer", [(None, None)])
    def test_invalid(
        self,
        fe_type: Union[str, None],
        fe_transfer: Union[str, None],
    ) -> None:
        with pytest.raises(ValueError):
            FeatureExtractor(fe_type, fe_transfer)


class TestOpenSMILE:
    @pytest.mark.parametrize(
        "feature_set, functionals, expected",
        [
            ("ComParE_2016", False, (65, 96)),
            ("ComParE_2016", True, (6373, 1)),
            ("eGeMAPSv02", False, (25, 96)),
            ("eGeMAPSv02", True, (88, 1)),
        ],
    )
    def test_opensmile(
        self,
        feature_set: str,
        functionals: bool,
        expected: Tuple[int, int],
    ) -> None:
        x = torch.randn(16000)
        y = OpenSMILE(feature_set, 16000, functionals=functionals)(x)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert y.shape == torch.Size(
            (1,) + expected
        ), "Output shape should match the expected shape"
        assert y.dtype == torch.float32, "Output should be float32"

    def test_invalid(self) -> None:
        with pytest.raises(FileNotFoundError):
            OpenSMILE(feature_set="fizz", sample_rate=16000)


class TestSmartCompose:
    @pytest.mark.parametrize(
        "other",
        [
            T.Compose([]),
            AnyToTensor(),
            lambda x: x,
            [lambda x: x],
        ],
    )
    def test_add_valid(
        self,
        other: Union[T.Compose, AbstractTransform, Callable, List[Callable]],
    ) -> None:
        assert isinstance(
            SmartCompose([]) + other, SmartCompose
        ), "Adding a Compose should return a SmartCompose"

    def test_add_none(self) -> None:
        sc = SmartCompose([])
        assert sc is sc + None, "Adding None should return the same object"

    def test_add_invalid(self) -> None:
        with pytest.raises(TypeError):
            SmartCompose([]) + 1

    @pytest.mark.parametrize(
        "transforms, has_collate_fn",
        [([CutMix()], True), ([AnyToTensor()], False)],
    )
    def test_collate_fn(
        self,
        transforms: List[
            Union[T.Compose, AbstractTransform, Callable, List[Callable]]
        ],
        has_collate_fn: bool,
    ) -> None:
        sc = SmartCompose(transforms)

        class MockDataset:
            output_dim = 10

        assert (
            sc.get_collate_fn(MockDataset()) is not None
        ) == has_collate_fn, "Collate function should be present"

    def test_sorting_order(self) -> None:
        att1 = AnyToTensor(order=1)
        att2 = AnyToTensor(order=2)
        att3 = AnyToTensor(order=3)
        order = [att1, att2, att3]
        sc = SmartCompose([]) + att2 + att1 + att3
        assert sc.transforms == order, "Transforms should be sorted by order"

    def test_sorting_stability(self) -> None:
        att1 = AnyToTensor()
        att2 = AnyToTensor()
        att3 = AnyToTensor()
        order = [att2, att1, att3]
        sc = SmartCompose([]) + att2 + att1 + att3
        assert sc.transforms == order, "Transforms should not be sorted"

    def test_call(self) -> None:
        sc = SmartCompose([AnyToTensor(), CutMix(p=0)])
        x = torch.randn(3, 32, 32)
        y = sc(x, 0)
        assert torch.is_tensor(y), "Output should be a tensor"
        assert torch.allclose(x, y), "Output should match the input"


class TestTransformManager:
    @classmethod
    def setup_class(cls) -> None:
        cls.m = {"type": "image"}
        cls.d = {"type": "image"}

    def test_dictconfig(self) -> None:
        TransformManager(DictConfig(self.m), DictConfig(self.d))

    @pytest.mark.parametrize("subset", ["base", "train", "dev", "test"])
    def test_specific(self, subset: str) -> None:
        TransformManager(self.m, self.d)._specific(subset)

    def test_invalid_specific(self) -> None:
        with pytest.raises(ValueError):
            TransformManager(self.m, self.d)._specific("fizz")

    def test_filter_transforms(self) -> None:
        m = {
            "type": "image",
            "train": [
                {"autrainer.transforms.Normalize": None},
            ],
        }
        d = {
            "type": "image",
            "train": [
                "autrainer.transforms.AnyToTensor",
                {"autrainer.augmentations.CutMix": {"p": 0.5}},
                {
                    "autrainer.transforms.Normalize": {
                        "mean": [0.0],
                        "std": [1.0],
                    }
                },
            ],
        }
        train, _, _ = TransformManager(m, d).get_transforms()
        assert len(train.transforms) == 2, "Transforms should be filtered"

    def test_combine_transforms(self) -> None:
        m = {
            "type": "image",
            "train": [
                "autrainer.transforms.AnyToTensor",
                {
                    "autrainer.transforms.Normalize": {
                        "mean": [10],
                        "std": [20],
                    }
                },
            ],
        }
        d = {
            "type": "image",
            "train": [
                {
                    "autrainer.transforms.Normalize": {
                        "mean": [0.0],
                        "std": [1.0],
                    }
                },
            ],
        }
        train, _, _ = TransformManager(m, d).get_transforms()
        assert len(train.transforms) == 2, "Transforms should be combined"
        norm = train.transforms[1]
        assert norm.mean == [10] and norm.std == [
            20
        ], "Transform should be overridden"

    @pytest.mark.parametrize(
        "model_type, dataset_type, valid",
        [
            ("image", "image", True),
            ("grayscale", "grayscale", True),
            ("raw", "raw", True),
            ("tabular", "tabular", True),
            ("image", "grayscale", True),
            ("grayscale", "image", True),
            ("image", "tabular", False),
            ("image", "raw", False),
            ("tabular", "raw", False),
            ("tabular", "image", False),
            ("raw", "image", False),
            ("raw", "tabular", False),
        ],
    )
    def test_valid_model_dataset_combination(
        self,
        model_type: str,
        dataset_type: str,
        valid: bool,
    ) -> None:
        tm = TransformManager({"type": model_type}, {"type": dataset_type})
        if valid:
            tm._match_model_dataset()
        else:
            with pytest.raises(ValueError):
                tm._match_model_dataset()

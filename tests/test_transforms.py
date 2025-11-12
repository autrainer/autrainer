from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Type, Union
import warnings

import numpy as np
from omegaconf import DictConfig
from PIL import Image
import pytest
import torch

from autrainer.augmentations import AbstractAugmentation, CutMix
from autrainer.core.structs import DataBatch, DataItem
from autrainer.datasets import AbstractDataset, ToyDataset
from autrainer.transforms import (
    AbstractTransform,
    AnyToTensor,
    Expand,
    FeatureExtractor,
    GrayscaleToRGB,
    ImageToFloat,
    Normalize,
    NumpyToTensor,
    OpenSMILE,
    PannMel,
    RandomCrop,
    Resample,
    Resize,
    RGBAToRGB,
    RGBToGrayscale,
    ScaleRange,
    SmartCompose,
    SpectToImage,
    SquarePadCrop,
    Standardizer,
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
    (ImageToFloat, {}, (3, 32, 32), (3, 32, 32)),
    (Normalize, {"mean": [0.0], "std": [1.0]}, (3, 32, 32), (3, 32, 32)),
    (RandomCrop, {"size": 8000, "axis": -1}, (1, 16000), (1, 8000)),
    (
        Resample,
        {"current_sr": 16000, "target_sr": 44100},
        (1, 16000),
        (1, 44100),
    ),
    (Resize, {"height": 64, "width": 128}, (3, 32, 32), (3, 64, 128)),
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
        ("transform", "params", "input_shape", "output_shape"),
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
            x = torch.randint(0, 256, input_shape, dtype=torch.uint8)
        device = x.device
        item = DataItem(x, 0, 0)
        transformed = instance(item)
        assert isinstance(transformed, DataItem), (
            "Transformed item should be of type DataItem."
        )
        assert isinstance(transformed.features, torch.Tensor), (
            "Transformed features should be of type torch.Tensor."
        )
        assert transformed.features.shape == output_shape, (
            "Output shape should match the expected shape."
        )
        assert transformed.features.device == device, (
            "Output device should match the input device."
        )

    def test_repr(self) -> None:
        assert repr(AnyToTensor()) == "AnyToTensor(order=-100)", (
            "Representation should match the expected string"
        )


class TestAnyToTensor:
    def test_tensor(self) -> None:
        x = DataItem(torch.randn(3, 32, 32), 0, 0)
        y = AnyToTensor()(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.dtype == torch.float32, "Output should be float32"
        assert y.features.device == x.features.device, (
            "Output device should match the input device"
        )

    def test_numpy(self) -> None:
        x = DataItem(np.random.rand(3, 32, 32).astype(np.float32), 0, 0)
        y = AnyToTensor()(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == torch.Size([3, 32, 32]), (
            "Output shape should match input"
        )
        assert y.features.dtype == torch.float32, "Output should be float32"
        assert y.features.device == torch.device("cpu"), "Output should be on CPU"

    def test_pil(self) -> None:
        x = Image.fromarray(np.random.rand(32, 32, 3).astype(np.uint8))
        x = DataItem(x, 0, 0)
        y = AnyToTensor()(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == torch.Size([3, 32, 32]), (
            "Output shape should match input"
        )
        assert y.features.dtype == torch.uint8, "Output should be uint8"
        assert y.features.device == torch.device("cpu"), "Output should be on CPU"

    def test_wrong_input(self) -> None:
        with pytest.raises(TypeError, match="must be a 'torch.Tensor'"):
            AnyToTensor()(DataItem(0, 0, 0))


class TestNumpyToTensor:
    def test_numpy(self) -> None:
        x = DataItem(np.random.rand(3, 32, 32).astype(np.float32), 0, 0)
        y = NumpyToTensor()(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == torch.Size([3, 32, 32]), (
            "Output shape should match input"
        )
        assert y.features.dtype == torch.float32, "Output should be float32"
        assert y.features.device == torch.device("cpu"), "Output should be on CPU"


class TestPannMel:
    def test_spectrogram(self) -> None:
        x = DataItem(torch.randn(1, 16000 * 3), 0, 0)
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
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == torch.Size([1, 301, 64]), (
            "Output shape should match input"
        )
        assert y.features.dtype == torch.float32, "Output should be float32"
        assert y.features.device == torch.device("cpu"), "Output should be on CPU"


class TestResize:
    @pytest.mark.parametrize(
        ("height", "width", "expected"),
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
        x = DataItem(torch.randn(3, 32, 32), 0, 0)
        if not expected:
            with pytest.raises(ValueError, match="must be specified"):
                Resize(height=height, width=width)
        else:
            r = Resize(height=height, width=width)
            assert r(x).features.shape == expected, (
                "Output shape should match the expected shape"
            )


class TestSquarePadCrop:
    @pytest.mark.parametrize(
        ("mode", "input_shape", "expected"),
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
        x = DataItem(torch.randn(*input_shape), 0, 0)
        if not expected:
            with pytest.raises(ValueError, match="Invalid mode"):
                SquarePadCrop(mode=mode)
        else:
            assert SquarePadCrop(mode=mode)(x).features.shape == expected, (
                "Output shape should match the expected shape"
            )


class TestScaleRange:
    @pytest.mark.parametrize("range", [(0, 1), (-1, 1), (0, 255)])
    def test_range_uint8(self, range: Tuple[int, int]) -> None:
        self._test_range(
            torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8),
            range,
        )

    @pytest.mark.parametrize("range", [(0, 1), (-1, 1), (0, 255)])
    def test_range_float32(self, range: Tuple[int, int]) -> None:
        self._test_range(torch.randn(3, 32, 32), range)

    def test_range_zeros(self) -> None:
        x = DataItem(torch.zeros(3, 32, 32), 0, 0)
        y = ScaleRange((0, 1))(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == torch.Size([3, 32, 32]), (
            "Output shape should match input"
        )
        assert y.features.dtype == torch.float32, "Output should be float32"
        assert y.features.min() == 0, "Min value should be 0"
        assert y.features.max() == 0, "Max value should be 0"

    @pytest.mark.parametrize("range", [(0, 0), (-1.5, -1.5)])
    def test_range_invalid(self, range: Tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="lower bound of 'range' must be"):
            ScaleRange(range)

    @pytest.mark.parametrize("range", [(0,), (1, 2, 3)])
    def test_range_len(self, range: Tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="Expected 'range' to be a list"):
            ScaleRange(range)

    def _test_range(self, x: torch.Tensor, range: Tuple[int, int]) -> None:
        y = ScaleRange(range)(DataItem(x, 0, 0))
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == torch.Size([3, 32, 32]), (
            "Output shape should match input"
        )
        assert y.features.dtype == torch.float32, "Output should be float32"
        assert y.features.min() == range[0], "Min value should match the range"
        assert y.features.max() == range[1], "Max value should match the range"


class TestImageToFloat:
    @pytest.mark.parametrize(
        "data",
        [
            torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8),
            torch.randn(3, 32, 32),
        ],
    )
    def test_image_to_float(self, data: torch.Tensor) -> None:
        x = DataItem(data, 0, 0)
        y = ImageToFloat()(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == x.features.shape, "Output shape should match input"
        assert y.features.dtype == torch.float32, "Output should be float32"


class TestNormalize:
    @pytest.mark.parametrize(
        "data",
        [
            torch.randn(4, 3, 32, 32, dtype=torch.float32),
            torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8),
        ],
    )
    def test_invalid_normalize(self, data: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="Unsupported feature dimensions"):
            Normalize(mean=[0.0, 1.0], std=[1.0])(DataItem(data, 0, 0))

    @pytest.mark.parametrize(
        "data",
        [
            torch.randn(64, dtype=torch.float32),
            torch.randn(4, 64, dtype=torch.float32),
            torch.randn(1, 64, 101, dtype=torch.float32),
            torch.randn(3, 32, 32, dtype=torch.float32),
            torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8),
        ],
    )
    def test_normalize(self, data: torch.Tensor) -> None:
        x = DataItem(data, 0, 0)
        y = Normalize(mean=[0.0], std=[1.0])(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == data.shape, "Output shape should match input"
        assert y.features.dtype == torch.float32, "Output should be float32"


class TestFeatureExtractor:
    @pytest.mark.parametrize(
        ("fe_type", "fe_transfer", "input_shape", "expected"),
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
        x = DataItem(torch.randn(*input_shape), 0, 0)
        y = fe(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == torch.Size(expected), (
            "Output shape should match the expected shape"
        )

    @pytest.mark.parametrize(("fe_type", "fe_transfer"), [(None, None)])
    def test_invalid(
        self,
        fe_type: Union[str, None],
        fe_transfer: Union[str, None],
    ) -> None:
        with pytest.raises(ValueError, match="must be provided"):
            FeatureExtractor(fe_type, fe_transfer)


class TestOpenSMILE:
    @pytest.mark.parametrize(
        ("feature_set", "functionals", "lld_deltas", "expected"),
        [
            ("ComParE_2016", False, False, (65, 96)),
            ("ComParE_2016", False, True, (130, 96)),
            ("ComParE_2016", True, False, (6373,)),
            ("eGeMAPSv02", False, False, (25, 96)),
            ("eGeMAPSv02", False, True, (25, 96)),
            ("eGeMAPSv02", True, False, (88,)),
        ],
    )
    def test_opensmile(
        self,
        feature_set: str,
        functionals: bool,
        lld_deltas: bool,
        expected: Tuple[int, int],
    ) -> None:
        x = DataItem(torch.randn(16000), 0, 0)
        y = OpenSMILE(
            feature_set, 16000, functionals=functionals, lld_deltas=lld_deltas
        )(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert y.features.shape == torch.Size(expected), (
            "Output shape should match the expected shape"
        )
        assert y.features.dtype == torch.float32, "Output should be float32"

    def test_invalid(self) -> None:
        with pytest.raises(FileNotFoundError):
            OpenSMILE(feature_set="fizz", sample_rate=16000)


class TestStandardizer:
    def _mock_dataset(
        self,
        feature_shape: List[int],
        transform: SmartCompose,
    ) -> AbstractDataset:
        return ToyDataset(
            task="classification",
            size=100,
            num_targets=10,
            feature_shape=feature_shape,
            dev_split=0.2,
            test_split=0.1,
            seed=0,
            dtype="uint8",
            metrics=["autrainer.metrics.Accuracy"],
            tracking_metric="autrainer.metrics.Accuracy",
            train_transform=transform,
        )

    @pytest.mark.parametrize("subset", ["fizz", "buzz", "jazz"])
    def test_invalid_subset(self, subset: str) -> None:
        with pytest.raises(ValueError, match="Invalid subset"):
            Standardizer(subset=subset)

    @pytest.mark.parametrize(
        ("mean", "std"),
        [
            ([0.0], [1.0]),
            ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]),
            ([10, 20, 30], [1, 2, 3]),
        ],
    )
    def test_precomputed(self, mean: List[float], std: List[float]) -> None:
        t1 = Standardizer(mean=mean, std=std)
        t2 = Normalize(mean=mean, std=std)
        x = DataItem(torch.randn(3, 32, 32), 0, 0)
        assert torch.allclose(t1(deepcopy(x)).features, t2(deepcopy(x)).features), (
            "Output should match Normalize."
        )

    def test_invalid_call(self) -> None:
        t = Standardizer()
        with pytest.raises(ValueError, match="transform not initialized"):
            t(DataItem(torch.randn(3, 32, 32), 0, 0))

    def test_invalid_setup(self) -> None:
        with pytest.raises(ValueError, match="Unsupported data dimensions"):
            self._mock_dataset([4, 3, 32, 32], SmartCompose([Standardizer()]))

    def test_setup(self) -> None:
        dataset = self._mock_dataset(
            [3, 32, 32],
            SmartCompose([Standardizer()]),
        )
        dataset.train_transform.setup(dataset)

    def test_setup_multiple_setup(self) -> None:
        s = Standardizer()
        dataset = self._mock_dataset([3, 32, 32], SmartCompose([s]))
        dataset.train_transform.setup(dataset)
        m1, s1 = s.mean, s.std
        dataset.train_transform.setup(dataset)
        m2, s2 = s.mean, s.std
        assert m1 == m2, "Mean should not change."
        assert s1 == s2, "Std should not change."

    def test_transform_order(self) -> None:
        sr = ScaleRange((10, 20), order=0)
        s1 = Standardizer(order=-1)
        s2 = Standardizer(order=1)
        ds_preceding = self._mock_dataset([3, 32, 32], SmartCompose([sr, s1]))
        ds_succeeding = self._mock_dataset([3, 32, 32], SmartCompose([sr, s2]))
        ds_preceding.train_transform.setup(ds_preceding)
        ds_succeeding.train_transform.setup(ds_succeeding)
        assert s1.mean != s2.mean, "Transforms should have different means."
        assert s1.std != s2.std, "Transforms should have different stds."

    def test_transform_order_precomputed(self) -> None:
        sr = ScaleRange((10, 20), order=0)
        s1 = Standardizer([1, 2, 3], [1, 2, 3], order=-1)
        s2 = Standardizer([1, 2, 3], [1, 2, 3], order=1)
        ds_preceding = self._mock_dataset([3, 32, 32], SmartCompose([sr, s1]))
        ds_succeeding = self._mock_dataset([3, 32, 32], SmartCompose([sr, s2]))
        ds_preceding.train_transform.setup(ds_preceding)
        ds_succeeding.train_transform.setup(ds_succeeding)
        assert s1.mean == s2.mean, "Transforms should have the same means."
        assert s1.std == s2.std, "Transforms should have the same stds."


class TestSmartCompose:
    @pytest.mark.parametrize(
        "other",
        [
            AnyToTensor(),
            SmartCompose([AnyToTensor()]),
            [AnyToTensor(), CutMix()],
        ],
    )
    def test_add_valid(
        self,
        other: Union[SmartCompose, AbstractTransform, List[AbstractTransform]],
    ) -> None:
        assert isinstance(SmartCompose([]) + other, SmartCompose), (
            "Adding a Compose should return a SmartCompose"
        )

    def test_add_none(self) -> None:
        sc = SmartCompose([])
        assert sc is sc + None, "Adding None should return the same object"

    def test_add_invalid(self) -> None:
        with pytest.raises(TypeError):
            SmartCompose([]) + 1

    @pytest.mark.parametrize(
        ("transforms", "has_collate_fn"),
        [([CutMix()], True), ([AnyToTensor()], False)],
    )
    def test_collate_fn(
        self,
        transforms: List[AbstractTransform],
        has_collate_fn: bool,
    ) -> None:
        sc = SmartCompose(transforms)

        class MockDataset:
            output_dim = 10

            @property
            def default_collate_fn(self) -> Callable:
                return DataBatch.collate

        assert sc.get_collate_fn(MockDataset()) is not None, (
            "Collate function should be present"
        )
        assert (
            sc.get_collate_fn(MockDataset()) == DataBatch.collate
        ) != has_collate_fn, f"Collate function should be default: {not has_collate_fn}"

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
        x = DataItem(torch.randn(3, 32, 32), 0, 0)
        y = SmartCompose([AnyToTensor(), CutMix(p=0)])(x)
        assert torch.is_tensor(y.features), "Output should be a tensor"
        assert torch.allclose(x.features, y.features), "Output should match the input"

    def test_setup(self) -> None:
        dataset = ToyDataset(
            task="classification",
            size=100,
            num_targets=10,
            feature_shape=[3, 32, 32],
            dev_split=0.2,
            test_split=0.1,
            seed=0,
            dtype="uint8",
            metrics=["autrainer.metrics.Accuracy"],
            tracking_metric="autrainer.metrics.Accuracy",
            train_transform=SmartCompose([AnyToTensor(), Standardizer(), CutMix(p=0)]),
        )
        dataset.train_transform.setup(dataset)


class TestTransformManager:
    @classmethod
    def setup_class(cls) -> None:
        cls.m = {"type": "image"}
        cls.d = {"type": "image"}

    def test_dictconfig(self) -> None:
        TransformManager(DictConfig(self.m), DictConfig(self.d))

    @pytest.mark.parametrize("subset", ["base", "train", "dev", "test"])
    def test_build(self, subset: str) -> None:
        TransformManager(self.m, self.d)._build(subset)

    @pytest.mark.parametrize("subset", ["base", "train", "dev", "test"])
    def test_filter_transforms(self, subset: str) -> None:
        m = {
            "type": "image",
            subset: [
                {"autrainer.transforms.Normalize": None},
            ],
        }
        d = {
            "type": "image",
            subset: [
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
        t = TransformManager(m, d).get_transforms()
        t = {"train": t[0], "dev": t[1], "test": t[2]}
        assert len(t.get(subset, t["train"]).transforms) == 2, (
            "Transforms should be filtered"
        )

    @pytest.mark.parametrize(
        ("sub1", "sub2", "count"),
        [
            ("base", "base", 2),
            ("base", "train", 3),
            ("train", "base", 3),
            ("train", "train", 2),
        ],
    )
    def test_override_transforms(
        self,
        sub1: str,
        sub2: str,
        count: int,
    ) -> None:
        m = {
            "type": "image",
            sub1: [
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
            sub2: [
                {
                    "autrainer.transforms.Normalize": {
                        "mean": [0.0],
                        "std": [1.0],
                    }
                },
            ],
        }
        train, _, _ = TransformManager(m, d).get_transforms()
        assert len(train.transforms) == count, "Transforms should be combined"

    @pytest.mark.parametrize(
        ("sub1", "sub2", "count"),
        [
            ("base", "base", 1),
            ("base", "train", 2),
            ("train", "base", 2),
            ("train", "train", 1),
        ],
    )
    def test_remove_transforms(self, sub1: str, sub2: str, count: int) -> None:
        m = {
            "type": "image",
            sub1: [
                "autrainer.transforms.AnyToTensor",
                {"autrainer.transforms.Normalize": None},
            ],
        }
        d = {
            "type": "image",
            sub2: [
                {
                    "autrainer.transforms.Normalize": {
                        "mean": [0.0],
                        "std": [1.0],
                    }
                },
            ],
        }
        train, _, _ = TransformManager(m, d).get_transforms()
        assert len(train.transforms) == count, "Should remove the transform."

    @pytest.mark.parametrize(
        ("sub1", "sub2", "count"),
        [
            ("base", "base", 3),
            ("base", "train", 4),
            ("train", "base", 4),
            ("train", "train", 3),
        ],
    )
    def test_replace_tag_transforms(
        self,
        sub1: str,
        sub2: str,
        count: int,
    ) -> None:
        m = {
            "type": "image",
            sub1: [
                "autrainer.transforms.AnyToTensor",
                {
                    "autrainer.transforms.Normalize@Tag": {
                        "mean": [0.0],
                        "std": [1.0],
                    }
                },
            ],
        }
        d = {
            "type": "image",
            sub2: [
                {
                    "autrainer.transforms.Normalize@Tag": {
                        "mean": [100],
                        "std": [100],
                    }
                },
                {
                    "autrainer.transforms.Normalize": {
                        "mean": [0.0],
                        "std": [1.0],
                    }
                },
            ],
        }
        train, _, _ = TransformManager(m, d).get_transforms()
        assert len(train.transforms) == count, "Transforms should be combined."

    @pytest.mark.parametrize(
        ("sub1", "sub2", "count"),
        [
            ("base", "base", 2),
            ("base", "train", 3),
            ("train", "base", 3),
            ("train", "train", 2),
        ],
    )
    def test_remove_tag_transforms(
        self,
        sub1: str,
        sub2: str,
        count: int,
    ) -> None:
        m = {
            "type": "image",
            sub1: [
                "autrainer.transforms.AnyToTensor",
                {"autrainer.transforms.Normalize@Tag": None},
            ],
        }
        d = {
            "type": "image",
            sub2: [
                {
                    "autrainer.transforms.Normalize@Tag": {
                        "mean": [100],
                        "std": [200],
                    }
                },
                {
                    "autrainer.transforms.Normalize": {
                        "mean": [0.0],
                        "std": [1.0],
                    }
                },
            ],
        }
        train, _, _ = TransformManager(m, d).get_transforms()
        assert len(train.transforms) == count, "Normalize@Tag should be removed."

    @pytest.mark.parametrize(
        ("model_type", "dataset_type", "valid"),
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
            with pytest.raises(ValueError, match="does not match"):
                tm._match_model_dataset()

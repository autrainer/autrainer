from typing import TYPE_CHECKING, List, Optional
import warnings

from audtorch import transforms as AT
from matplotlib.pyplot import get_cmap
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from torchaudio import transforms as TT
import torchlibrosa
from torchvision import transforms as T
from transformers import (
    ASTFeatureExtractor,
    AutoFeatureExtractor,
    Wav2Vec2FeatureExtractor,
    WhisperFeatureExtractor,
)

from autrainer.core.structs import AbstractDataItem

from .abstract_transform import AbstractTransform
from .smart_compose import SmartCompose
from .utils import to_numpy, to_tensor


try:
    import opensmile

    OPENSMILE_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENSMILE_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from autrainer.datasets import AbstractDataset
    from autrainer.datasets.utils import DatasetWrapper


FE_MAPPINGS = {
    "AST": {"fe": ASTFeatureExtractor, "padding": "max_length"},
    "Whisper": {"fe": WhisperFeatureExtractor, "padding": "max_length"},
    "W2V2": {"fe": Wav2Vec2FeatureExtractor, "padding": "longest"},
    None: {"fe": AutoFeatureExtractor, "padding": "max_length"},
}


class AnyToTensor(AbstractTransform):
    def __init__(self, order: int = -100) -> None:
        """Convert a numpy array, torch tensor, or a PIL image to a torch
        tensor.

        Args:
            order: The order of the transform in the pipeline. Defaults to -100.
        """
        super().__init__(order=order)
        self._convert_pil = T.ToTensor()

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        if isinstance(item.features, torch.Tensor):
            return item
        if isinstance(item.features, np.ndarray):
            item.features = to_tensor(item.features)
        elif isinstance(item.features, Image.Image):
            item.features = self._convert_pil(item.features) * 255
            item.features = item.features.to(torch.uint8)
        else:
            raise TypeError(
                "Input must be a 'torch.Tensor', 'np.ndarray', or "
                "'PIL.Image.Image', but got "
                f"'{item.features.__class__.__name__}'"
            )
        return item


class NumpyToTensor(AbstractTransform):
    def __init__(self, order: int = -100) -> None:
        """Convert a numpy array to a torch tensor.

        Args:
            order: The order of the transform in the pipeline. Defaults to -100.
        """
        super().__init__(order=order)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = to_tensor(item.features)
        return item


class SpectToImage(AbstractTransform):
    def __init__(
        self,
        height: int,
        width: int,
        cmap: str = "magma",
        order: int = -90,
    ) -> None:
        """Convert a spectrogram in the range [0, 1] to a 3-channel uint8 image
        in the range [0, 255] using a specific colormap.

        Note: The transform should be used in combination with
        `~autrainer.transforms.ImageToFloat` to convert the image back to the
        range [0, 1] for training.

        Args:
            height: The height of the image.
            width: The width of the image.
            cmap: The colormap to use. Defaults to "magma".
            order: The order of the transform in the pipeline. Defaults to -90.
        """
        super().__init__(order=order)
        self.height = height
        self.width = width
        self.cmap = cmap
        self._cmap = get_cmap(cmap)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        if item.features.dtype == torch.uint8:
            item.features = item.features / 255
        im = item.features.numpy().squeeze()
        im -= im.min()
        im /= im.max()
        im = (
            Image.fromarray(np.uint8(self._cmap(im)[..., :3] * 255))
            .transpose(Image.FLIP_TOP_BOTTOM)
            .resize((self.width, self.height))
        )
        item.features = torch.from_numpy(np.array(im)).transpose(0, 2)
        return item


class PannMel(AbstractTransform):
    def __init__(
        self,
        window_size: int,
        hop_size: int,
        sample_rate: int,
        fmin: int,
        fmax: int,
        mel_bins: int,
        ref: float,
        amin: float,
        top_db: int,
        order: int = -90,
    ) -> None:
        """Create a log-Mel spectrogram from an audio signal analogous to
        the Pretrained Audio Neural Networks (PANN) implementation.

        For more information, see:
        https://doi.org/10.48550/arXiv.1912.10211

        Args:
            window_size: The size of the window.
            hop_size: Hop length.
            sample_rate: The sample rate of the audio signal.
            fmin: The minimum frequency.
            fmax: The maximum frequency.
            mel_bins: The number of mel bins.
            ref: The reference amplitude.
            amin: The minimum amplitude.
            top_db: The top decibel.
            order: The order of the transform in the pipeline. Defaults to -90.
        """
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.mel_bins = mel_bins
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        super().__init__(order=order)
        self._spectrogram = torchlibrosa.stft.Spectrogram(
            n_fft=self.window_size,
            win_length=self.window_size,
            hop_length=self.hop_size,
        )
        self._mel = torchlibrosa.stft.LogmelFilterBank(
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n_mels=self.mel_bins,
            n_fft=self.window_size,
            ref=self.ref,
            amin=self.amin,
            top_db=self.top_db,
        )

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = self._mel(self._spectrogram(item.features)).squeeze(1)
        return item


class Resize(AbstractTransform):
    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        interpolation: str = "bilinear",
        antialias: bool = True,
        order: int = -90,
    ) -> None:
        """Resize an image to a specific height and width.
        If "Any" is provided for the height or width, the other dimension will
        be used as the size, creating a square image.

        Args:
            height: The target height. If set to "Any", the width will be used
                as the size. Defaults to 64.
            width: The target width. If set to "Any", the height will be used
                as the size. Defaults to 64.
            interpolation: The interpolation method to use. Defaults to
                'bilinear'.
            antialias: Whether to use antialiasing. Defaults to True.
            order: The order of the transform in the pipeline. Defaults to -90.
        """
        super().__init__(order=order)
        self.width = width
        self.height = height
        self.antialias = antialias
        self.interpolation = interpolation
        if self.width == "Any" and self.height == "Any":
            raise ValueError("At least one of 'height' or 'width' must be specified.")
        if self.height == "Any":
            self._size = self.width
        elif self.width == "Any":
            self._size = self.height
        else:
            self._size = (self.height, self.width)
        self._resize = T.Resize(
            size=self._size,
            interpolation=getattr(T.InterpolationMode, self.interpolation.upper()),
            antialias=self.antialias,
        )

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = self._resize(item.features)
        return item


class SquarePadCrop(AbstractTransform):
    def __init__(self, mode: str = "pad", order: int = -91) -> None:
        """Pad or crop an image to make it square.
        If the image is padded, the padding will be added to the shorter
        sides as black pixels.

        Args:
            mode: The mode to use in ["pad", "crop"]. Defaults to "pad".
            order: The order of the transform in the pipeline. Defaults to -91.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        super().__init__(order=order)
        self.mode = mode
        if self.mode == "pad":
            self._mode_fn = max
        elif self.mode == "crop":
            self._mode_fn = min
        else:
            raise ValueError(f"Invalid mode '{self.mode}'.")

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        m = self._mode_fn(item.features.shape[-2:])
        item.features = T.functional.center_crop(item.features, m)
        return item


class ScaleRange(AbstractTransform):
    def __init__(
        self,
        range: List[float] = None,
        order: int = 90,
    ) -> None:
        """Scale a tensor to a specific target range.

        Args:
            range: The range to scale the tensor to. If None, it will be set to
            [0.0, 1.0]. Defaults to None.
            order: The order of the transform in the pipeline. Defaults to 90.
        """
        if range is None:
            range = [0.0, 1.0]
        super().__init__(order=order)
        if len(range) != 2:
            raise ValueError(
                f"Expected 'range' to be a list of length 2 but got {range}."
            )
        if range[0] == range[1]:
            raise ValueError(
                "The lower bound of 'range' must be different from the upper "
                f"bound but got {range}."
            )
        self.range = sorted(range)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        m, M, (r0, r1) = item.features.min(), item.features.max(), self.range
        if m == M:
            item.features = torch.full_like(item.features, m)
            return item
        item.features = (item.features - m) / (M - m)
        item.features = item.features * (r1 - r0) + r0
        return item


class ImageToFloat(AbstractTransform):
    def __init__(self, order: int = 90) -> None:
        """Transform a uint8 image in the range [0, 255] to a float32 image in
        the range [0.0, 1.0].

        Args:
            order: The order of the transform in the pipeline. Defaults to 90.
        """
        super().__init__(order=order)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        if item.features.dtype == torch.uint8:
            item.features = item.features.float() / 255
        return item


class Normalize(AbstractTransform):
    def __init__(
        self,
        mean: List[float],
        std: List[float],
        order: int = 95,
    ) -> None:
        """Normalize a tensor with a mean and standard deviation.

        Args:
            mean: The mean to use for normalization.
            std: The standard deviation to use for normalization.
            order: The order of the transform in the pipeline. Defaults to 95.
        """
        super().__init__(order=order)
        self.mean = mean
        self.std = std
        self._mean = torch.as_tensor(mean)
        self._std = torch.as_tensor(std)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        # reshapes mean and std to match the data dimensions and broadcast
        # the values along the correct axes
        views = {
            3: (-1, 1, 1),  # 3D images: normalize along the channel axis
            2: (-1, 1),  # 2D audio: normalize along the channel axis
            1: (-1,),  # 1D tabular data: normalize along the feature axis
        }

        for dim, axes in views.items():  # noqa: B007
            if item.features.ndim == dim:
                break
        else:
            raise ValueError(f"Unsupported feature dimensions: {item.features.shape}")

        mean, std = self._mean.view(*axes), self._std.view(*axes)
        item.features = item.features.to(torch.float32).sub(mean).div(std)
        return item


class Standardizer(AbstractTransform):
    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        subset: str = "train",
        order: int = -99,
    ) -> None:
        """Standardize a dataset by calculating the mean and standard deviation
        of the data and applying the normalization.

        The mean and standard deviation are automatically calculated from the
        data in the specified subset.

        Note: The transform is applied at the specified order in the pipeline
        such that any preceding transforms are applied before the mean and
        standard deviation are calculated.

        Args:
            mean: The mean to use for normalization. If None, the mean will be
                calculated from the data. Defaults to None.
            std: The standard deviation to use for normalization. If None, the
                standard deviation will be calculated from the data. Defaults
                to None.
            subset: The subset to use for calculating the mean and standard
                deviation. Must be one of ["train", "dev", "test"]. Defaults to
                "train".
            order: The order of the transform in the pipeline. Defaults to -99.
        """
        super().__init__(order=order)
        self._validate_subset(subset)
        self.subset = subset
        self.mean = mean
        self.std = std
        self._transform = self._init_transform()

    def _validate_subset(self, subset: str) -> None:
        if subset not in {"train", "dev", "test"}:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of 'train', 'dev', or 'test'."
            )

    def _init_transform(self) -> Optional[AbstractTransform]:
        if self.mean and self.std:
            return Normalize(self.mean, self.std, self.order)
        return None

    def _collect_data(self, data: "AbstractDataset") -> torch.Tensor:
        ds: DatasetWrapper = getattr(data, f"{self.subset}_dataset")
        original_transform = ds.transform

        idx = original_transform.transforms.index(self)  # thanks to audobject
        preceding = SmartCompose(original_transform.transforms[:idx])
        ds.transform = preceding

        data = torch.stack([x.features for x in ds]).to(torch.float32)
        ds.transform = original_transform
        return data

    def setup(self, data: "AbstractDataset") -> None:
        if self.mean and self.std:  # if already computed for the current data
            return

        # select all dimensions except the dimension along which to normalize
        views = {
            4: (0, 2, 3),  # 3D images: normalize along the channel axis
            3: (0, 2),  # 2D audio: normalize along the channel axis
            2: (0,),  # 1D tabular data: normalize along the feature axis
        }
        collected = self._collect_data(data)
        for dim, axes in views.items():  # noqa: B007
            if collected.ndim == dim:
                break
        else:
            raise ValueError(f"Unsupported data dimensions: {collected.shape}")

        self.mean = collected.mean(dim=axes).tolist()
        self.std = collected.std(dim=axes).tolist()
        self._transform = self._init_transform()

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        if self._transform is None:
            raise ValueError("Standardize transform not initialized.")
        return self._transform(item)


class FeatureExtractor(AbstractTransform):
    def __init__(
        self,
        fe_type: Optional[str] = None,
        fe_transfer: Optional[str] = None,
        sampling_rate: int = 16000,
        order: int = -80,
    ) -> None:
        """Extract features from an audio signal using a feature extractor
        from the Hugging Face Transformers library.

        Args:
            fe_type: The class of feature extractor to use in ["AST", "Whisper",
                "W2V2", None]. If None, the AutoFeatureExtractor will be used.
                Defaults to None.
            fe_transfer: The name of a pretrained feature extractor to use.
                If None, the feature extractor will be initialized with default
                values. Defaults to None.
            sampling_rate: The sampling rate of the audio signal. Defaults to
                16000.
            order: The order of the transform in the pipeline. Defaults to -80.

        Raises:
            ValueError: If neither 'fe_type' nor 'fe_transfer' is provided.
        """
        super().__init__(order=order)
        if fe_type is None and fe_transfer is None:
            raise ValueError("Either 'fe_type' or 'fe_transfer' must be provided.")
        self.fe_type = fe_type
        self.fe_transfer = fe_transfer
        self.sampling_rate = sampling_rate
        fe_class = FE_MAPPINGS[self.fe_type]["fe"]
        padding = FE_MAPPINGS[self.fe_type]["padding"]

        if self.fe_transfer is not None:
            feature_extractor = fe_class.from_pretrained(self.fe_transfer)
        else:
            feature_extractor = fe_class()
            extractor_dict = {k: repr(v) for k, v in feature_extractor.__dict__.items()}
            warnings.warn(
                f"{fe_class.__name__} "
                "initialized with default values:\n"
                f"{OmegaConf.to_yaml(extractor_dict)}",
                stacklevel=2,
            )

        def extract_features(signal: np.ndarray) -> torch.Tensor:
            if len(signal.shape) == 2:
                signal = signal.mean(0)
            extracted = feature_extractor(
                signal,
                sampling_rate=self.sampling_rate,
                padding=padding,
                return_tensors="pt",
            )
            return extracted[list(extracted.keys())[0]][0]

        self._extract_features = extract_features

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = self._extract_features(item.features.numpy())
        return item


class Expand(AbstractTransform):
    def __init__(
        self,
        size: int,
        method: str = "pad",
        axis: int = -2,
        order: int = -85,
    ) -> None:
        """Expand a tensor along a specific axis to a specific size, ensuring
        it is at least the specified size.
        If the tensor is smaller than the target size, it will be padded
        with zeros.
        If the tensor is larger than the target size, it will not be cropped.

        Args:
            size: The target size.
            method: The method to use in ["pad", "replicate"]. Defaults to
                "pad".
            axis: The axis along which to expand. Defaults to -2.
            order: The order of the transform in the pipeline. Defaults to -85.
        """
        super().__init__(order=order)
        self.size = size
        self.method = method
        self.axis = axis
        self._expand = AT.Expand(size=self.size, method=self.method, axis=self.axis)
        self._to_tensor = AnyToTensor()

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = self._expand(item.features)
        return self._to_tensor(item)


class RandomCrop(AbstractTransform):
    def __init__(
        self,
        size: int,
        method: str = "pad",
        axis: int = -2,
        fix_randomization: bool = False,
        order: int = -90,
    ) -> None:
        """Randomly crop a tensor along a specific axis to a specific size,
        ensuring it is the specified size.

        If the tensor is larger than the target size, it will be randomly
        cropped along the specified axis.

        If the tensor is smaller than the target size, it will be padded.

        Args:
            size: The target size.
            method: Padding method to use if the tensor is smaller than the
                target size in ["pad", "replicate"]. Defaults to "pad".
            axis: The axis along which to crop. Defaults to -2.
            fix_randomization: Whether to fix the randomization. Defaults to
                False.
            order: The order of the transform in the pipeline. Defaults to -90.
        """
        super().__init__(order=order)
        self.size = size
        self.method = method
        self.axis = axis
        self.fix_randomization = fix_randomization
        self._random_crop = AT.RandomCrop(
            size=self.size,
            method=self.method,
            axis=self.axis,
            fix_randomization=self.fix_randomization,
        )
        self._to_tensor = AnyToTensor()

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = self._random_crop(item.features)
        return self._to_tensor(item)


class StereoToMono(AbstractTransform):
    def __init__(self, order: int = -95) -> None:
        """Convert a stereo audio signal to mono by taking the mean of the
        first dimension.

        Args:
            order: The order of the transform in the pipeline. Defaults to -95.
        """
        super().__init__(order=order)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = item.features.mean(0, keepdim=True)
        return item


class RGBAToRGB(AbstractTransform):
    """Convert an RGBA image to RGB by dropping the alpha channel.

    Args:
        order: The order of the transform in the pipeline. Defaults to -95.
    """

    def __init__(self, order: int = -95) -> None:
        super().__init__(order=order)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = item.features[:3]
        return item


class RGBToGrayscale(AbstractTransform):
    def __init__(self, order: int = 100) -> None:
        """Convert an RGB or RGBA image to a grayscale image.

        For the conversion to grayscale, the
        luminance is calculated in line with the implementation in
        torchvision.transforms.Grayscale:
        Y = 0.2989 * R + 0.587 * G + 0.114 * B.

        Args:
            order: The order of the transform in the pipeline. Defaults to 100.
        """
        super().__init__(order=order)
        self._to_grayscale = T.Grayscale()

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = self._to_grayscale(item.features[:3])
        return item


class GrayscaleToRGB(AbstractTransform):
    def __init__(self, order: int = 100) -> None:
        """Convert a grayscale image to an RGB image.

        Args:
            order: The order of the transform in the pipeline. Defaults to 100.
        """
        super().__init__(order=order)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = to_tensor(AT.Upmix(3, axis=0)(to_numpy(item.features)))
        return item


class Resample(AbstractTransform):
    def __init__(
        self,
        current_sr: int,
        target_sr: int,
        order: int = -95,
    ) -> None:
        """Resample an audio signal to a target sample rate.

        Args:
            current_sr: The current sample rate.
            target_sr: The target sample rate.
            order: The order of the transform in the pipeline. Defaults to -95.
        """
        self.current_sr = current_sr
        self.target_sr = target_sr
        super().__init__(order=order)
        self._resample = TT.Resample(orig_freq=self.current_sr, new_freq=self.target_sr)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = self._resample(item.features)
        return item


class OpenSMILE(AbstractTransform):
    def __init__(
        self,
        feature_set: str,
        sample_rate: int,
        functionals: bool = False,
        lld_deltas: bool = False,
        order: int = -80,
    ) -> None:
        """Extract features from an audio signal using openSMILE.

        The openSMILE library must be installed to use this transform.
        To install the required extras, run:
        'pip install autrainer[opensmile]'.

        .. warning::
            Setting `lld_deltas` to `True`
            will extract and concatenate
            :py:attr:`opensmile.FeatureLevel.LowLevelDescriptors_Deltas` to the
            :py:attr:`opensmile.FeatureLevel.LowLevelDescriptors` features.
            However, :py:mod:`opensmile` returns *longer* sequences
            for the delta features. To match to the original length,
            we drop the first and last frame of the delta features.
            This may result in misalignment, as we are not sure
            how padding is handled in :py:mod:`opensmile`.

        Args:
            feature_set: The feature set to use.
            sample_rate: The sample rate of the audio signal.
            functionals: Whether to use functionals. Defaults to False.
            lld_deltas: Whether to additionally compute deltas. Only relevant
                for LLDs. Defaults to False.
            order: The order of the transform in the pipeline. Defaults to -80.

        Raises:
            ImportError: If openSMILE is not available.
        """
        if not OPENSMILE_AVAILABLE:
            raise ImportError(
                "openSMILE is not available for feature extraction. Install "
                "the required extras with 'pip install autrainer[opensmile]'."
            )  # pragma: no cover
        super().__init__(order=order)
        self.feature_set = feature_set
        self.sample_rate = sample_rate
        self.functionals = functionals
        self.lld_deltas = lld_deltas
        self.smile_de = None
        if self.functionals:
            self.smile = opensmile.Smile(self.feature_set)
        else:
            self.smile = opensmile.Smile(
                self.feature_set,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
            if self.lld_deltas and self.feature_set != "eGeMAPSv02":
                self.smile_de = opensmile.Smile(
                    self.feature_set,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
                )

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        it = self.smile.process_signal(item.features, self.sample_rate)
        feats = torch.from_numpy(self.smile.to_numpy(it)).squeeze()
        if self.smile_de is not None:
            data_de = self.smile_de.process_signal(item.features, self.sample_rate)
            data_de = torch.from_numpy(self.smile.to_numpy(data_de)).squeeze()[:, 1:-1]
            feats = torch.cat((feats, data_de), axis=-2)
        item.features = feats
        return item

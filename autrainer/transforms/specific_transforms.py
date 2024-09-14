from typing import List, Optional, Union
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

from .abstract_transform import AbstractTransform
from .utils import _to_numpy, _to_tensor


try:
    import opensmile

    OPENSMILE_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENSMILE_AVAILABLE = False  # pragma: no cover


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

    def __call__(
        self,
        data: Union[torch.Tensor, np.ndarray, Image.Image],
    ) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return _to_tensor(data)
        elif isinstance(data, Image.Image):
            return (self._convert_pil(data) * 255).to(torch.uint8)
        else:
            raise TypeError(
                "Input must be a 'torch.Tensor', 'np.ndarray' "
                f"or 'PIL.Image.Image' but got '{data.__class__.__name__}'"
            )


class NumpyToTensor(AbstractTransform):
    def __init__(self, order: int = -100) -> None:
        """Convert a numpy array to a torch tensor.

        Args:
            order: The order of the transform in the pipeline. Defaults to -100.
        """
        super().__init__(order=order)

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _to_tensor(data)


class SpectToImage(AbstractTransform):
    def __init__(
        self,
        height: int,
        width: int,
        cmap: str = "magma",
        order: int = -90,
    ):
        """Convert a spectrogram to a 3-channel image based on a colormap.

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

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.dtype == torch.uint8:
            data = data.float()
        data = data.numpy().squeeze()
        data -= data.min()
        data /= data.max()
        data = np.uint8(self._cmap(data)[..., :3] * 255)
        im = Image.fromarray(data)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        im = im.resize((self.width, self.height))
        return torch.from_numpy(np.array(im)).transpose(0, 2).to(torch.uint8)


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
    ):
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

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._mel(self._spectrogram(data)).squeeze(1)


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
            raise ValueError(
                "At least one of 'height' or 'width' must be specified."
            )
        if self.height == "Any":
            self._size = self.width
        elif self.width == "Any":
            self._size = self.height
        else:
            self._size = (self.height, self.width)
        self._resize = T.Resize(
            size=self._size,
            interpolation=getattr(
                T.InterpolationMode, self.interpolation.upper()
            ),
            antialias=self.antialias,
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._resize(data)


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

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return T.functional.center_crop(data, self._mode_fn(data.shape[-2:]))


class ScaleRange(AbstractTransform):
    def __init__(
        self,
        range: List[float] = [0.0, 1.0],
        order: int = 90,
    ) -> None:
        """Scale a tensor to a specific target range.

        Args:
            range: The range to scale the tensor to. Defaults to [0.0, 1.0].
            order: The order of the transform in the pipeline. Defaults to 90.
        """
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

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data_min, data_max = data.min(), data.max()
        if data_min == data_max:
            return torch.full_like(data, data_min)
        data = (data - data_min) / (data_max - data_min)
        return data * (self.range[1] - self.range[0]) + self.range[0]


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
        self._normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.dtype == torch.uint8:
            data = data.float()
        return self._normalize(data)


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
            raise ValueError(
                "Either 'fe_type' or 'fe_transfer' must be provided."
            )
        self.fe_type = fe_type
        self.fe_transfer = fe_transfer
        self.sampling_rate = sampling_rate
        fe_class = FE_MAPPINGS[self.fe_type]["fe"]
        padding = FE_MAPPINGS[self.fe_type]["padding"]

        if self.fe_transfer is not None:
            feature_extractor = fe_class.from_pretrained(self.fe_transfer)
        else:
            feature_extractor = fe_class()
            extractor_dict = {
                k: repr(v) for k, v in feature_extractor.__dict__.items()
            }
            warnings.warn(
                f"{fe_class.__name__} "
                "initialized with default values:\n"
                f"{OmegaConf.to_yaml(extractor_dict)}"
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

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._extract_features(data.numpy())


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
        self._expand = AT.Expand(
            size=self.size, method=self.method, axis=self.axis
        )
        self._to_tensor = AnyToTensor()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._to_tensor(self._expand(data))


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

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._to_tensor(self._random_crop(data))


class StereoToMono(AbstractTransform):
    def __init__(self, order: int = -95) -> None:
        """Convert a stereo audio signal to mono by taking the mean of the
        first dimension.

        Args:
            order: The order of the transform in the pipeline. Defaults to -95.
        """
        super().__init__(order=order)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data.mean(0, keepdim=True)


class RGBAToRGB(AbstractTransform):
    """Convert an RGBA image to RGB by dropping the alpha channel.

    Args:
        order: The order of the transform in the pipeline. Defaults to -95.
    """

    def __init__(self, order: int = -95) -> None:
        super().__init__(order=order)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[:3]


class RGBToGrayscale(AbstractTransform):
    def __init__(self, order: int = 100) -> None:
        """Convert an RGB image to a grayscale image.

        For the conversion to grayscale, the
        luminance is calculated in line with the implementation in
        torchvision.transforms.Grayscale:
        Y = 0.2989 * R + 0.587 * G + 0.114 * B.

        Args:
            order: The order of the transform in the pipeline. Defaults to 100.
        """
        super().__init__(order=order)
        self._to_grayscale = T.Grayscale()

    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        image_grs = self._to_grayscale(image)
        return image_grs


class RGBAToGrayscale(AbstractTransform):
    def __init__(self, order: int = -95) -> None:
        """Convert an RGBA image to grayscale by dropping the alpha channel
        and converting to grayscale.

        For the conversion to grayscale, the
        luminance is calculated in line with the implementation in
        torchvision.transforms.Grayscale:
        Y = 0.2989 * R + 0.587 * G + 0.114 * B.


        Args:
            order: The order of the transform in the pipeline. Defaults to -95.
        """
        super().__init__(order=order)
        self._to_grayscale = T.Grayscale()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._to_grayscale(data[:3])


class GrayscaleToRGB(AbstractTransform):
    def __init__(self, order: int = 100) -> None:
        """Convert a grayscale image to an RGB image.

        Args:
            order: The order of the transform in the pipeline. Defaults to 100.
        """
        super().__init__(order=order)

    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        image = _to_numpy(image)
        image_rgb = AT.Upmix(3, axis=0)(image)
        return _to_tensor(image_rgb)


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
        self._resample = TT.Resample(
            orig_freq=self.current_sr, new_freq=self.target_sr
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._resample(data)


class OpenSMILE(AbstractTransform):
    def __init__(
        self,
        feature_set: str,
        sample_rate: int,
        functionals: bool = False,
        order: int = -80,
    ) -> None:
        """Extract features from an audio signal using openSMILE.

        The openSMILE library must be installed to use this transform.
        To install the required extras, run:
        'pip install autrainer[opensmile]'.

        Args:
            feature_set: The feature set to use.
            sample_rate: The sample rate of the audio signal.
            functionals: Whether to use functionals. Defaults to False.
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
        if self.functionals:
            self.smile = opensmile.Smile(self.feature_set)
        else:
            self.smile = opensmile.Smile(
                self.feature_set,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = self.smile.process_signal(data.numpy(), self.sample_rate)
        return torch.from_numpy(self.smile.to_numpy(data))

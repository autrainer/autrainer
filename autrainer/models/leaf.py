from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract_model import AbstractModel
from .timm_wrapper import TimmModel


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    from speechbrain.lobes.features import Leaf as LeafSb


class LEAFNet(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        leaf_filters: int = 40,
        kernel_size: int = 25,
        stride: float = 0.0625,
        window_stride: int = 10,
        padding_kernel_size: int = 25,
        sample_rate: int = 16000,
        min_freq: int = 60,
        max_freq: int = 7800,
        efficientnet_type: str = "efficientnet_b0",
        mode: str = "interspeech",
        initialization: str = "mel",
        generator_seed: int = 42,
        transfer: bool = False,
    ) -> None:
        """EfficientNet with LEAF-Is frontend.
        Used to reproduce work from:
        https://www.isca-archive.org/interspeech_2023/meng23c_interspeech.html

        Also see original LEAF and PCEN papers (c.f. speechbrain).

        We take and slightly adapt the LEAF frontend implementation from:
        https://github.com/Hanyu-Meng/Adapting-LEAF

        Args:
            output_dim: Output dimension.
            leaf_filters: Number of LEAF filterbanks to train. Defaults to 40.
            kernel_size: Size of kernels applied by LEAF (in ms).
                Defaults to 25.
            stride: Stride of LEAF (in ms). Defaults to 0.0625.
            window_stride: Stride of lowpass filter (in ms). Defaults to 10.
            padding_kernel_size: Size of lowpass filter (in ms).
                Defaults to 25.
            sample_rate: Used to compute LEAF params. Defaults to 16000.
            min_freq: Minimum freq analyzed by LEAF. Defaults to 60.
            max_freq: Maximum freq analyzed by LEAF. Defaults to 7800.
            efficientnet_type: EfficientNet type to use from timm.
                Defaults to "efficientnet_b0".
            mode: Implementation according to "interspeech" paper (Meng et al.)
                or "speech_brain". Defaults to "interspeech".
            initialization: Filterbank initialisation in ["mel", "bark",
                "linear-constant", "constant", "uniform", "zeros"].
                Defaults to "mel".
            generator_seed: Seed for random generator. Defaults to 42.
            transfer: Whether to use EfficientNet weights from ImageNet.
                Defaults to False.

        Raises:
            ValueError: If efficientnet_type is not supported.
            ValueError: If mode is not supported.
        """
        super().__init__(output_dim)
        # convert LEAF params from ms to samples
        self.mode = mode
        self.initialization = initialization
        self.leaf_filters = leaf_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.generator_seed = generator_seed
        self.kernel_size = kernel_size
        self.stride = stride
        self.window_stride = window_stride
        self.padding_kernel_size = padding_kernel_size
        self.efficientnet_type = efficientnet_type
        self.transfer = transfer
        kernel_size_sample = int(sample_rate * kernel_size / 1000)
        kernel_size_sample += 1 - (kernel_size % 2)  # make odd

        # convert pooling params from ms to samples
        if mode == "interspeech":
            self.leaf = LeafIs(
                n_filters=leaf_filters,
                min_freq=min_freq,
                max_freq=max_freq,
                sample_rate=sample_rate,
                window_len=padding_kernel_size,
                window_stride=window_stride,
            )

        elif mode == "speech_brain":
            self.leaf = LeafSb(
                out_channels=leaf_filters,
                in_channels=1,
                sample_rate=sample_rate,
                min_freq=min_freq,
                use_pcen=True,
                learnable_pcen=True,
                use_legacy_complex=True,
                skip_transpose=True,
            )
        else:
            raise ValueError(
                "Only options 'interspeech' and 'speech_brain' are available"
                f" for mode, but got mode='{mode}'."
            )
        self._initialise_filterbank()
        if not self.efficientnet_type.startswith("efficientnet_"):
            raise ValueError(
                "Only EfficientNet models are supported, but got"
                f" efficientnet_type='{efficientnet_type}'."
            )
        self.classifier = TimmModel(
            output_dim=self.output_dim,
            timm_name=self.efficientnet_type,
            transfer=self.transfer,
            in_chans=1,
        )

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaf(x)

    def forward(self, x: torch.Tensor):
        x = self.leaf(x)
        x = x.unsqueeze(1)
        x = self.classifier(x)
        return x

    def _initialise_filterbank(self) -> None:
        if self.initialization == "mel":
            return
        elif self.initialization == "linear-constant":
            center_frequencies = torch.linspace(
                self.min_freq, self.max_freq, self.leaf_filters
            )
            bandwidths = torch.ones((self.leaf_filters,)) * np.exp(
                (np.log(self.max_freq) - np.log(self.min_freq)) / 2.5
                + np.log(self.min_freq)
            )
        elif self.initialization == "zeros":
            center_frequencies = torch.zeros((self.leaf_filters,))
            bandwidths = torch.ones((self.leaf_filters,)) * np.exp(
                np.log(self.max_freq - self.min_freq) / 2
            )
        elif self.initialization == "constant":
            center_frequencies = torch.ones((self.leaf_filters,)) * np.exp(
                (np.log(self.max_freq) - np.log(self.min_freq)) / 2
                + np.log(self.min_freq)
            )
            bandwidths = torch.ones((self.leaf_filters,)) * np.exp(
                (np.log(self.max_freq) - np.log(self.min_freq)) / 2.5
                + np.log(self.min_freq)
            )
        elif self.initialization == "uniform":
            generator = torch.Generator()
            generator.manual_seed(self.generator_seed)
            center_frequencies = (
                torch.rand((self.leaf_filters,), generator=generator)
                * (self.max_freq - self.min_freq)
                + self.min_freq
            )
            center_frequencies = torch.sort(center_frequencies).values
            # Estimation
            bandwidths = (
                torch.rand((self.leaf_filters,), generator=generator)
                * (self.max_freq - self.min_freq)
                / 10
                + self.min_freq
            )
        elif self.initialization == "bark":
            center_frequencies, bandwidths = bark_scale_filterbank(
                self.min_freq, self.max_freq, self.leaf_filters
            )
        else:
            raise ValueError(
                "Only options 'mel', 'linear-constant', 'constant', 'uniform', "
                f"'bark', and 'zeros' are available for initialization "
                f", but got initialization='{self.initialization}'."
            )

        # adjustment for sample_rate
        center_frequencies *= 2 * np.pi / self.sample_rate
        # bandwiths to sigmas
        bandwidths = (self.sample_rate / 2.0) / bandwidths

        if self.mode == "interspeech":
            center_freqs_param = "filterbank.center_freqs"
            bandwith_param = "filterbank.bandwidths"
            self.leaf.state_dict()[center_freqs_param].copy_(
                center_frequencies
            )
            self.leaf.state_dict()[bandwith_param].copy_(bandwidths)
        elif self.mode == "speech_brain":
            filterbank_param = "complex_conv.kernel"
            self.leaf.state_dict()[filterbank_param][:, 0].copy_(
                center_frequencies
            )
            self.leaf.state_dict()[filterbank_param][:, 1].copy_(bandwidths)


def hz_to_bark(
    f: Union[int, float, np.ndarray],
) -> Union[int, float, np.ndarray]:
    """Convert frequency from Hz to Bark scale according to Traunmüller.

    Args:
        f: Frequency in Hz.

    Returns:
        Frequency in Bark.
    """
    b = 26.81 * f / (1960 + f) - 0.53
    return b


def bark_to_hz(
    b: Union[int, float, np.ndarray],
) -> Union[int, float, np.ndarray]:
    """Approximate conversion from Bark to Hz (inverse of the above).
    Ajdusted centers for reconversion according to Traunmüller.

    Args:
        b: Frequency in Bark.

    Returns:
        Frequency in Hz.
    """

    z_1 = b[b < 2] + 0.15 * (2 - b[b < 2])
    z_2 = b[b >= 2]
    z_2 = z_2[z_2 <= 20.1]
    z_3 = b[b > 20.1] + 0.22 * (b[b > 20.1] - 20.1)
    z = np.hstack((z_1, z_2, z_3))
    f = 1960 * (z + 0.53) / (26.28 - z)
    return f  # 600 * np.sinh(z / 6)


def bark_scale_filterbank(
    min_freq: Union[int, float],
    max_freq: Union[int, float],
    num_bands: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate center frequencies and bandwidths for a Bark scale filterbank.

    Args:
        min_freq: Minimum frequency in Hz.
        max_freq: Maximum frequency in Hz.
        num_bands: Number of bands.

    Returns:
        Tuple of center frequencies and bandwidths.
    """
    # Convert min and max frequencies to Bark scale
    min_bark = hz_to_bark(min_freq)
    max_bark = hz_to_bark(max_freq)

    # Evenly distribute bands in Bark scale
    bark_centers = np.linspace(
        min_bark, max_bark + (max_bark - min_bark) / num_bands, num_bands + 1
    )

    # Convert center frequencies back to Hz
    center_freqs = bark_to_hz(bark_centers)
    bandwidths = np.diff(center_freqs)

    return torch.Tensor(center_freqs[:-1]), torch.Tensor(bandwidths)


def mel_filter_params(
    n_filters: int,
    min_freq: float,
    max_freq: float,
    sample_rate: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytically calculate the center frequencies and sigmas of a mel
    filter bank.

    Args:
        n_filters: Number of filters for the filterbank.
        min_freq: Minimum cutoff for the frequencies.
        max_freq: Maximum cutoff for the frequencies.
        sample_rate: Sample rate to use for the calculation.

    Returns:
        Center frequencies and sigmas.
    """
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    peaks_mel = torch.linspace(min_mel, max_mel, n_filters + 2)
    peaks_hz = 700 * (torch.expm1(peaks_mel / 1127))
    center_freqs = peaks_hz[1:-1] * (2 * np.pi / sample_rate)
    bandwidths = peaks_hz[2:] - peaks_hz[:-2]
    sigmas = (sample_rate / 2.0) / bandwidths
    return center_freqs, sigmas


def gabor_filters(
    size: int, center_freqs: torch.Tensor, sigmas: torch.Tensor
) -> torch.Tensor:
    """Calculate a gabor function from given center frequencies and bandwidths
    that can be used as kernel/filters for an 1D convolution.

    Args:
        size: Kernel/filter size.
        center_freqs: Center frequencies.
        sigmas: Bandwidths.

    Returns:
        Kernel/filter that can be used for an 1D convolution.
    """
    t = torch.arange(-(size // 2), (size + 1) // 2, device=center_freqs.device)
    denominator = 1.0 / (np.sqrt(2 * np.pi) * sigmas)
    gaussian = torch.exp(torch.outer(1.0 / (2.0 * sigmas**2), -(t**2)))
    sinusoid = torch.exp(1j * torch.outer(center_freqs, t))
    return denominator[:, np.newaxis] * sinusoid * gaussian


def gauss_windows(size: int, sigmas: torch.Tensor) -> torch.Tensor:
    """Calculate a gaussian lowpass function from given bandwidths that can be
    used as kernel/filter for an 1D convolution.

    Args:
        size: Kernel/filter size.
        sigmas: Bandwidths.

    Returns:
        Kernel/filter that can be used for an 1D convolution.
    """
    t = torch.arange(0, size, device=sigmas.device)
    numerator = t * (2 / (size - 1)) - 1
    return torch.exp(-0.5 * (numerator / sigmas[:, np.newaxis]) ** 2)


class GaborFilterbank(nn.Module):
    def __init__(
        self,
        n_filters: int,
        min_freq: float,
        max_freq: float,
        sample_rate: int,
        filter_size: int,
        pool_size: int,
        pool_stride: int,
        pool_init: float = 0.4,
    ) -> None:
        """Torch module that functions as a gabor filterbank.

        Initializes n_filters center frequencies and bandwidths that are based
        on a mel filterbank. The parameters are used to calculate Gabor filters
        for a 1D convolution over the input signal. The squared modulus is
        taken from the results. To reduce the temporal resolution a gaussian
        lowpass filter is calculated from pooling_widths, which are used to
        perform a pooling operation. The center frequencies, bandwidths and
        pooling_widths are learnable parameters.

        Args:
            n_filters: Number of filters.
            min_freq: Minimum frequency for the mel filterbank initialization.
            max_freq: Maximum frequency for the mel filterbank initialization.
            sample_rate: Sample rate for the mel filterbank initialization.
            filter_size: Size of the kernels/filters for gabor convolution.
            pool_size: Size of the kernels/filters for pooling convolution.
            pool_stride: Stride of the pooling convolution.
            pool_init: Initial value for the gaussian lowpass function.
                Defaults to 0.4.
        """
        super(GaborFilterbank, self).__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        center_freqs, bandwidths = mel_filter_params(
            n_filters, min_freq, max_freq, sample_rate
        )
        self.center_freqs = nn.Parameter(center_freqs)
        self.bandwidths = nn.Parameter(bandwidths)
        self.pooling_widths = nn.Parameter(
            torch.full((n_filters,), float(pool_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute filters
        center_freqs = self.center_freqs.clamp(min=0.0, max=np.pi)
        z = np.sqrt(2 * np.log(2)) / np.pi
        bandwidths = self.bandwidths.clamp(min=4 * z, max=self.filter_size * z)
        filters = gabor_filters(self.filter_size, center_freqs, bandwidths)
        filters = torch.cat((filters.real, filters.imag), dim=0).unsqueeze(1)
        # convolve with filters
        x = F.conv1d(x, filters, padding=self.filter_size // 2)
        # compute squared modulus
        x = x**2
        x = x[:, : self.n_filters] + x[:, self.n_filters :]
        # compute pooling windows
        pooling_widths = self.pooling_widths.clamp(
            min=2.0 / self.pool_size, max=0.5
        )
        windows = gauss_windows(self.pool_size, pooling_widths).unsqueeze(1)
        # apply temporal pooling
        x = F.conv1d(
            x,
            windows,
            stride=self.pool_stride,
            padding=self.filter_size // 2,
            groups=self.n_filters,
        )
        return x


class PCEN(nn.Module):
    def __init__(
        self,
        num_bands: int,
        s: float = 0.025,
        alpha: float = 1.0,
        delta: float = 1.0,
        r: float = 1.0,
        eps: float = 1e-6,
        learn_logs: bool = False,
        clamp: Optional[float] = None,
    ) -> None:
        """Trainable PCEN (Per-Channel Energy Normalization) layer.

        .. math::
            Y = (\\frac{X}{(\\epsilon + M)^\\alpha} + \\delta)^r - \\delta^r
            M_t = (1 - s) M_{t - 1} + s X_t

        Args:
            num_bands: Number of frequency bands (before last input dimension).
            s: Initial value for :math:`s`.
            alpha: Initial value for :math:`alpha`
            delta: Initial value for :math:`delta`
            r: Initial value for :math:`r`
            eps: Value for :math:`eps`
            learn_logs: If false-ish, instead of learning the logarithm of each
                parameter (as in the PCEN paper), learn the inverse of :math:`r`
                and all other parameters directly (as in the LEAF paper).
            clamp: If given, clamps the input to the given minimum value before
                applying PCEN.
        """
        super(PCEN, self).__init__()
        if learn_logs:
            # learns logarithm of each parameter
            s = np.log(s)
            alpha = np.log(alpha)
            delta = np.log(delta)
            r = np.log(r)
        else:
            # learns inverse of r, and all other parameters directly
            r = 1.0 / r
        self.learn_logs = learn_logs
        self.s = nn.Parameter(torch.full((num_bands,), float(s)))
        self.alpha = nn.Parameter(torch.full((num_bands,), float(alpha)))
        self.delta = nn.Parameter(torch.full((num_bands,), float(delta)))
        self.r = nn.Parameter(torch.full((num_bands,), float(r)))
        self.eps = torch.as_tensor(eps)
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clamp if needed
        if self.clamp is not None:
            x = x.clamp(min=self.clamp)

        # prepare parameters
        if self.learn_logs:
            # learns logarithm of each parameter
            s = self.s.exp()
            alpha = self.alpha.exp()
            delta = self.delta.exp()
            r = self.r.exp()
        else:
            # learns inverse of r, and all other parameters directly
            s = self.s
            alpha = self.alpha.clamp(max=1)
            delta = self.delta.clamp(min=0)  # unclamped in original LEAF impl.
            r = 1.0 / self.r.clamp(min=1)
        # broadcast over channel dimension
        alpha = alpha[:, np.newaxis]
        delta = delta[:, np.newaxis]
        r = r[:, np.newaxis]

        # compute smoother
        smoother = [x[..., 0]]  # initialize the smoother with the first frame
        for frame in range(1, x.shape[-1]):
            smoother.append((1 - s) * smoother[-1] + s * x[..., frame])
        smoother = torch.stack(smoother, -1)

        # stable reformulation due to Vincent Lostanlen; original formula was:
        # return (input / (self.eps + smoother)**alpha + delta)**r - delta**r
        smoother = torch.exp(
            -alpha * (torch.log(self.eps) + torch.log1p(smoother / self.eps))
        )
        return (x * smoother + delta) ** r - delta**r


class LeafIs(nn.Module):
    def __init__(
        self,
        n_filters: int = 40,
        min_freq: float = 60.0,
        max_freq: float = 7800.0,
        sample_rate: int = 16000,
        window_len: float = 25.0,
        window_stride: float = 10.0,
        compression: Optional[torch.nn.Module] = None,
    ) -> None:
        """LEAF frontend, a learnable front-end that takes an audio waveform as
        input and outputs a learnable spectral representation.
        Initially approximates the computation of standard mel-filterbanks.

        Args:
            n_filters: Number of filters. Defaults to 40.
            min_freq: Minimum frequency. Defaults to 60.0.
            max_freq: Maximum frequency. Defaults to 7800.0.
            sample_rate: Sample Rate for filterbanc initialization.
                Defaults to 16000.
            window_len: Kernel/filter size of the convolutions in ms.
                Defaults to 25.0.
            window_stride: Stride used for the pooling convolution in ms.
                Defaults to 10.0.
            compression: Compression function. If None, PCEN is used.
                Defaults to None.
        """
        super(LeafIs, self).__init__()

        # convert window sizes from milliseconds to samples
        window_size = int(sample_rate * window_len / 1000)
        window_size += 1 - (window_size % 2)  # make odd
        window_stride = int(sample_rate * window_stride / 1000)

        self.filterbank = GaborFilterbank(
            n_filters,
            min_freq,
            max_freq,
            sample_rate,
            filter_size=window_size,
            pool_size=window_size,
            pool_stride=window_stride,
        )

        self.compression = (
            compression
            if compression
            else PCEN(
                n_filters,
                s=0.04,
                alpha=0.96,
                delta=2,
                r=0.5,
                eps=1e-12,
                learn_logs=False,
                clamp=1e-5,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.ndim < 3:
            x = x[:, np.newaxis]
        x = self.filterbank(x)
        x = self.compression(x)
        return x

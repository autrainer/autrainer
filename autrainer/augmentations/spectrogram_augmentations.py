from typing import Optional

import torch
import torchaudio.transforms as T
import torchvision.transforms.functional as F

from autrainer.core.structs import AbstractDataItem

from .abstract_augmentation import AbstractAugmentation
from .spectrogram_warp_utils import _sparse_image_warp


class GaussianNoise(AbstractAugmentation):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Add Gaussian noise to the input tensor with mean and standard
        deviation.

        Args:
            mean: The mean of the Gaussian noise. Defaults to 0.0.
            std: The standard deviation of the Gaussian noise. Defaults to 1.0.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        self.mean = mean
        self.std = std
        self._generator = torch.Generator()
        if generator_seed is not None:
            self._generator.manual_seed(generator_seed)

    def offset_generator_seed(self, offset: int) -> None:
        super().offset_generator_seed(offset)
        if self.generator_seed is not None:
            self._generator.manual_seed(self.generator_seed)

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        r = torch.randn(item.features.size(), generator=self._generator)
        item.features = item.features + r * self.std + self.mean
        return item


class TimeShift(AbstractAugmentation):
    def __init__(
        self,
        axis: int,
        time_steps: int = 0,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Shift the input tensor along the time axis.

        Args:
            axis: Time axis. If the image is torch Tensor, it is expected
                to have [C, H, W] shape, then H is assumed to be axis 0, and W
                is axis 1.
            time_steps: maximum time steps a tensor will shifted
                forward or backward. Defaults to 0.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        if time_steps < 0:
            raise ValueError(f"Time steps '{time_steps}' must be >= 0.")
        super().__init__(order, p, generator_seed)
        self.axis = axis
        self.time_steps = time_steps
        self._generator = torch.Generator()
        if generator_seed is not None:
            self._generator.manual_seed(generator_seed)

    def offset_generator_seed(self, offset: int) -> None:
        super().offset_generator_seed(offset)
        if self.generator_seed is not None:
            self._generator.manual_seed(self.generator_seed)

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        if self.time_steps == 0:
            return item

        t = torch.randint(
            -self.time_steps,
            self.time_steps,
            (1,),
            generator=self._generator,
        ).item()
        if self.axis == 1:
            t1, t2 = item.features[:, :, :-t], item.features[:, :, -t:]
        else:
            t1, t2 = item.features[:, :-t, :], item.features[:, -t:, :]
        item.features = torch.cat((t2, t1), dim=self.axis + 1)
        return item


class TimeMask(AbstractAugmentation):
    def __init__(
        self,
        time_mask: int,
        axis: int,
        replace_with_zero: bool = True,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Mask a random number of time steps.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            time_mask: maximum time steps in a tensor will be masked.
            axis: Time axis. If the image is torch Tensor, it is expected
                to have [C, H, W] shape, then H is assumed to be axis 0, and W
                is axis 1.
            replace_with_zero: Fill the mask either with a tensor mean, or 0's.
                Defaults to True.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        if time_mask < 0:
            raise ValueError(f"Time mask '{time_mask}' must be >= 0.")
        super().__init__(order, p, generator_seed)
        self._deterministic = False
        self.time_mask = time_mask
        self.axis = axis
        self.replace_with_zero = replace_with_zero
        self.masking = T.TimeMasking(time_mask_param=self.time_mask)

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        if self.time_mask == 0:
            return item
        if self.axis == 0:
            item.features = torch.rot90(item.features, 3, [1, 2])
            item.features = self.masking(item.features)
            item.features = torch.rot90(item.features, 1, [1, 2])
        else:
            item.features = self.masking(item.features)
        return item


class FrequencyMask(AbstractAugmentation):
    def __init__(
        self,
        freq_mask: int,
        axis: int,
        replace_with_zero: bool = True,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """Mask a random number of frequency steps.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        Args:
            freq_mask: maximum frequency steps in a tensor will be masked.
            axis: Frequency axis. If the image is torch Tensor, it is
                expected to have [C, H, W] shape, then H is assumed to be axis 0,
                and W is axis 1.
            replace_with_zero: Fill the mask either with a tensor mean, or 0's.
                Defaults to True.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        if freq_mask < 0:
            raise ValueError(f"Frequency mask '{freq_mask}' must be >= 0.")
        super().__init__(order, p, generator_seed)
        self._deterministic = False
        self.freq_mask = freq_mask
        self.axis = axis
        self.replace_with_zero = replace_with_zero
        self.masking = T.FrequencyMasking(freq_mask_param=self.freq_mask)

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        if self.freq_mask == 0:
            return item
        if self.axis == 1:
            item.features = torch.rot90(item.features, 3, [1, 2])
            item.features = self.masking(item.features)
            item.features = torch.rot90(item.features, 1, [1, 2])
        else:
            item.features = self.masking(item.features)
        return item


class TimeWarp(AbstractAugmentation):
    def __init__(
        self,
        axis: int,
        W: int = 10,  # noqa: N803
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """A random point along the time axis passing through the center of
        the image within the time steps (W, tau - W) is to be warped either to
        the left or right by a distance w chosen from a uniform distribution
        from 0 to the time warp parameter W along that line.

        Args:
            axis: Time axis. If the image is torch Tensor, it is expected
                to have [C, H, W] shape, then H is assumed to be axis 0, and W
                is axis 1.
            W: Bound for squishing/stretching. Defaults to 10.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        self.axis = axis
        self.W = W
        self._generator = torch.Generator()
        if generator_seed is not None:
            self._generator.manual_seed(generator_seed)

    def offset_generator_seed(self, offset: int) -> None:
        super().offset_generator_seed(offset)
        if self.generator_seed is not None:
            self._generator.manual_seed(self.generator_seed)

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        device = item.features.device
        if self.axis == 0:
            item.features = torch.rot90(item.features, 3, [1, 2])

        _, num_freq_channels, len_time = F.get_dimensions(item.features)

        # random point along the time axis
        pt = (len_time - 2 * self.W) * torch.rand(
            [1],
            dtype=torch.float,
            generator=self._generator,
        ) + self.W

        # source
        # control points on freq-axis
        src_ctr_pt_freq = torch.arange(0, num_freq_channels // 2)
        src_ctr_pt_time = (
            torch.ones_like(src_ctr_pt_freq) * pt
        )  # control points on time-axis
        src_ctr_pts = torch.stack((src_ctr_pt_freq, src_ctr_pt_time), dim=-1)
        src_ctr_pts = src_ctr_pts.float().to(device)

        # Destination
        w = (
            2 * self.W * torch.rand([1], dtype=torch.float, generator=self._generator)
            - self.W
        )  # distance
        dest_ctr_pt_freq = src_ctr_pt_freq
        dest_ctr_pt_time = src_ctr_pt_time + w
        dest_ctr_pts = torch.stack(
            (dest_ctr_pt_freq, dest_ctr_pt_time),
            dim=-1,
        )
        dest_ctr_pts = dest_ctr_pts.float().to(device)

        src_ctr_pt_locations = torch.unsqueeze(src_ctr_pts, 0)
        dest_ctr_pt_locations = torch.unsqueeze(dest_ctr_pts, 0)

        warped_spectro, _ = _sparse_image_warp(
            item.features,
            src_ctr_pt_locations,
            dest_ctr_pt_locations,
        )

        if self.axis == 0:
            warped_spectro = torch.rot90(warped_spectro, 1, [1, 2])
        item.features = warped_spectro
        return item


class SpecAugment(AbstractAugmentation):
    def __init__(
        self,
        time_mask: int = 10,
        freq_mask: int = 10,
        W: int = 50,  # noqa: N803
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        """SpecAugment augmentation.
        A combination of time warp, frequency masking, and time masking.

        Important: While the probability of applying the augmentation is
        deterministic if the generator_seed is set, the actual augmentation
        applied is not deterministic. This is because the internal random
        number generator of the augmentation is not seeded.

        For more information, see:
        https://arxiv.org/abs/1904.08779

        This implementation differs from PyTorch, as they apply TimeStrech
        instead of TimeWarp. For more information, see:
        https://pytorch.org/audio/master/tutorials/audio_feature_augmentation_tutorial.html#specaugment

        Args:
            time_mask: maximum time steps in a tensor will be masked.
                Defaults to 10.
            freq_mask: maximum frequency steps in a tensor will be masked.
                Defaults to 10.
            W: Bound for squishing/stretching the time axis. Defaults to 50.
            order: The order of the augmentation in the transformation pipeline.
                Defaults to 0.
            p: The probability of applying the augmentation. Defaults to 1.0.
            generator_seed: The initial seed for the internal random number
                generator drawing the probability. If None, the generator is
                not seeded. Defaults to None.
        """
        super().__init__(order, p, generator_seed)
        self._deterministic = False
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.W = W
        self._time_warp = TimeWarp(W=self.W, axis=0)
        self._freq_mask = FrequencyMask(
            freq_mask=self.freq_mask,
            replace_with_zero=True,
            axis=1,
        )
        self._time_mask = TimeMask(
            time_mask=self.time_mask,
            replace_with_zero=True,
            axis=0,
        )

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        item = self._time_warp(item)
        item = self._freq_mask(item)
        return self._time_mask(item)

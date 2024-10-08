import scipy.ndimage
import torch

from autrainer.transforms import AbstractTransform


class SpectMedianFilter(AbstractTransform):
    def __init__(self, size: int, order: int = 0) -> None:
        """Spectrogram median filter to remove noise.

        Args:
            size: Number of neighboring pixels to consider when filtering.
                Must be odd.
            order: The order of the transform in the pipeline. Larger means
                later in the pipeline. If multiple transforms have the same
                order, they are applied in the order they were added to the
                pipeline. Defaults to 0.
        """
        super().__init__(order=order)
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            scipy.ndimage.median_filter(
                x.cpu().numpy(),
                size=self.size,
            )
        ).to(x.device)

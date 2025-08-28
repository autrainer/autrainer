from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch

from .utils import assert_nonzero_frequency


if TYPE_CHECKING:  # pragma: no cover
    from autrainer.datasets import AbstractDataset


class MSELoss(torch.nn.MSELoss):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.MSELoss.forward`.

        Squeezes the model outputs along the last dimension to match the shape
        of the targets.
        Converts the targets to `float`.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        # TODO: is there a more elegant way to handle broadcasting
        return super().forward(x.squeeze(-1), y.float())


class WeightedMSELoss(MSELoss):
    weights_buffer: torch.Tensor

    def __init__(
        self,
        target_weights: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> None:
        """Wrapper for `torch.nn.MSELoss` with manual target weights intended
        for multi-target regression tasks.

        The target weights are automatically normalized to sum up to the number
        of targets.

        Args:
            target_weights: Dictionary with target weights corresponding to the
                target labels and their respective weights.
            **kwargs: Additional keyword arguments passed to
                `torch.nn.MSELoss`.
        """
        self.target_weights = target_weights
        super().__init__(**kwargs)

    def setup(self, data: "AbstractDataset") -> None:
        """Calculate the target weights based on the provided dictionary.

        Args:
            data: Instance of the dataset.
        """
        values = []

        if data.task != "mt-regression":
            raise ValueError(
                "`WeightedMSELoss` is only supported for multi-target regression tasks."
            )

        for target in data.target_transform.target:
            if target not in self.target_weights:
                raise ValueError(f"Missing target weight for target '{target}'.")
            values.append(self.target_weights[target])

        assert_nonzero_frequency(np.array(values), len(data.target_transform))
        weight = torch.tensor(values, dtype=torch.float32)
        weight = weight * len(weight) / weight.sum()
        self.register_buffer("weights_buffer", weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.MSELoss.forward` with manual target weights.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        return super().forward(x, y) * self.weights_buffer.expand_as(y)

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
from torch.nn.modules.loss import _Loss

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


class CCCLoss(_Loss):
    """Concordance correlation coefficient loss.

    A typical loss for dimensional speech emotion recognition
    (https://arxiv.org/abs/2203.07378).
    It is computed as 1 minus the concordance correlation coefficient,
    also known as Lin's correlation coefficient
    (https://en.wikipedia.org/wiki/Concordance_correlation_coefficient).
    It is a scaled version
    of the Pearson correlation coefficient.

    The correlation is computed over targets in a batch.
    It is recommended to use it with larger batch sizes
    than the standard :class:`~autrainer.criterions.MSELoss`.

    .. note::
        We are using biased estimators
        for the variance and standard deviation
        of the prediction and ground truth tensors
        in order to match the computation
        in :meth:`~audmetric.concordance_cc`
        which relies on `numpy`.
        In some cases,
        this may lead to different results
        than when using unbiased estimators.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computation of CCC loss.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """

        # squeeze needed in case of single-target regression
        # to avoid dimension projection
        y = y.squeeze()
        x = x.squeeze()
        mean_gt = torch.mean(y, axis=0)
        mean_pred = torch.mean(x, axis=0)
        var_gt = torch.var(y, axis=0, unbiased=False)
        var_pred = torch.var(x, axis=0, unbiased=False)
        v_pred = x - mean_pred
        v_gt = y - mean_gt
        corr = torch.sum(v_pred * v_gt, axis=0) / (
            torch.sqrt(torch.sum(v_pred**2, axis=0))
            * torch.sqrt(torch.sum(v_gt**2, axis=0))
        )
        sd_gt = torch.std(y, axis=0, unbiased=False)
        sd_pred = torch.std(x, axis=0, unbiased=False)
        numerator = 2 * corr * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        ccc = numerator / denominator
        ccc = ccc.mean()
        if ccc != ccc:
            # handle NaNs
            # these happen when denominator=0
            # i.e. when both pred and truth
            # are the same, constant vector
            ccc = torch.tensor(0)
        return 1 - ccc

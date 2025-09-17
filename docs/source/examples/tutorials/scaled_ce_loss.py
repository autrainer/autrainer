from typing import Any, Dict

import torch


class ScaledCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        scaling_factor: float = 1.0,
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Cross entropy loss with a scaling factor.

        Args:
            scaling_factor: Scaling factor for the loss.
            *args: Positional arguments passed to `torch.nn.CrossEntropyLoss`.
            **kwargs: Keyword arguments passed to `torch.nn.CrossEntropyLoss`.
        """
        super().__init__(*args, **kwargs)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y.long()
        return self.scaling_factor * super().forward(x, y)

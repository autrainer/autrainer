from typing import Callable, Tuple

import torch


class SomeOptimizer(torch.optim.Optimizer):
    def custom_step(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        criterion: torch.nn.Module,
        probabilities_fn: Callable,
    ) -> Tuple[float, torch.Tensor]:
        """Custom step function for the optimizer.

        Args:
            model: Model to be optimized.
            data: Batched input data.
            target: Batched target data.
            criterion: Loss function.
            probabilities_fn: Function to get probabilities from model outputs.

        Returns:
            Reduced loss over the batch and detached model outputs.
        """
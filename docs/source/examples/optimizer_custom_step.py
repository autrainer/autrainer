from typing import Callable, Tuple

import torch

from autrainer.core.structs import DataBatch
from autrainer.models import AbstractModel


class SomeOptimizer(torch.optim.Optimizer):
    def custom_step(
        self,
        model: AbstractModel,
        data: DataBatch,
        criterion: torch.nn.Module,
        probabilities_fn: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom step function for the optimizer.

        Args:
            model: The model to train.
            data: The data batch containing features, target, and potentially
                additional fields. The data batch is on the same
                device as the model. Additional fields are passed to the model
                as keyword arguments if they are present in the model's forward
                method.
            criterion: Loss function.
            probabilities_fn: Function to convert model outputs to
                probabilities.

        Returns:
            Tuple containing the non-reduced loss and model outputs.
        """

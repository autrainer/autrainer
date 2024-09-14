from typing import Tuple

import torch


class SomeOptimizer(torch.optim.Optimizer):
    def custom_step(
        self,
        model: torch.nn.Module,  # model
        data: torch.Tensor,  # batched input data
        target: torch.Tensor,  # batched target data
        criterion: torch.nn.Module,  # loss function
    ) -> Tuple[float, torch.Tensor]:
        loss = ...  # reduced loss over the batch
        outputs = ...  # detached model outputs
        return loss, outputs

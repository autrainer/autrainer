from typing import List

import torch
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmUpLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        """Linear warm-up learning rate scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of warmup steps.
            last_epoch: The index of last epoch. Defaults to -1.
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        return self.base_lrs

import math
from typing import List, Union

import torch

from autrainer.datasets.utils import AbstractTargetTransform


class LogTargetTransform(AbstractTargetTransform):
    def __init__(self, base: int = 10, eps: float = 1e-10) -> None:
        """Logarithmic target transform for regression tasks.

        Args:
            base: Base of the logarithm. Defaults to 10.
            eps: Small value to avoid taking the logarithm of zero.
                Defaults to 1e-10.
        """
        self.base = base
        self.eps = eps

    def encode(self, x: float) -> float:
        return math.log(x + self.eps, self.base)

    def decode(self, x: float) -> float:
        return math.pow(self.base, x) - self.eps

    def predict_batch(self, x: torch.Tensor) -> Union[List[float], float]:
        return x.squeeze().tolist()

    def majority_vote(self, x: List[float]) -> float:
        return sum(x) / len(x)

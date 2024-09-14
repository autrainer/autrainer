from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from autrainer.datasets import AbstractDataset


class ExampleLoss(torch.nn.modules.loss._Loss):
    def setup(self, data: "AbstractDataset") -> None: ...

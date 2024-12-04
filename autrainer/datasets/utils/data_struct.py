from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class Data:
    features: torch.Tensor
    target: Union[int, float, List[int], List[float]]
    index: int

    def to(self, device):
        self.features.to(device)
        self.target.to(device)
        self.index.to(device)

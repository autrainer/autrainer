from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class Data:
    features: torch.Tensor
    label: Union[int, float, List[int], List[float]]
    index: int

    def to(self, device):
        self.features.to(device)
        self.label.to(device)
        self.index.to(device)

from typing import Union

import numpy as np
import torch


def _to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    return data


def _to_tensor(data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data

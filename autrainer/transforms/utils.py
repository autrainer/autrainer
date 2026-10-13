import numpy as np
import torch


def to_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    return data


def to_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data

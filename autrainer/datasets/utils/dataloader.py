from collections.abc import Mapping
from typing import List

import numpy as np
import torch

from .data_struct import Data


def default_data_collator(features: List[Data]) -> Data:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in features])
        elif isinstance(v, np.ndarray):
            batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))
        else:
            batch[k] = torch.tensor([f[k] for f in features])

    return Data(**batch)

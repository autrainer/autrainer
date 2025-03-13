import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set a global seed for reproducibility for random, numpy, and torch.

    If CUDA is available, set the seed for CUDA and cuDNN as well.

    Args:
        seed: Seed to set.
    """
    if seed != seed % 2**32:
        raise ValueError(f"Seed must be between 0 and 2**32-1, got '{seed}'.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

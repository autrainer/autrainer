import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set a global seed for reproducibility for random, numpy, and torch.

    If CUDA is available, set the seed for CUDA and cuDNN as well.

    Args:
        seed: Seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

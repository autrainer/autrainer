import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set a global seed for random, numpy, and torch.

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
        torch.cuda.manual_seed_all(seed)


def _deterministic() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _nondeterministic() -> None:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def set_reproducibility(reproducible: bool) -> None:
    """Set the reproducibility of the training process.

    Args:
        reproducible: Whether to make the training process reproducible.
            If True, the training process will be deterministic and use only
            deterministic algorithms. If False, the training process may be
            non-deterministic and use non-deterministic algorithms.
    """
    _deterministic() if reproducible else _nondeterministic()

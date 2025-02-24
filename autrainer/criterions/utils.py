import numpy as np


def assert_nonzero_frequency(frequency: np.ndarray, num_targets: int) -> None:
    """Check if the frequency array contains non-zero values for all classes
    and has the correct length and therefore does not contain any missing
    (zero) class-frequencies.

    Args:
        frequency: Array with the frequency of each class.
        num_targets: Number of target classes.

    Raises:
        ValueError: If the frequency array contains zero values or has an
            incorrect length.
    """
    if (frequency == 0).any() or len(frequency) != num_targets:
        raise ValueError(
            "Balanced weighting requires a non-zero frequency for all classes."
        )

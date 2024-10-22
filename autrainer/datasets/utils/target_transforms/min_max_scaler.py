from typing import List, Union

import torch

from .abstract_target_transform import AbstractTargetTransform


class MinMaxScaler(AbstractTargetTransform):
    def __init__(self, minimum: float, maximum: float) -> None:
        """Minimum-Maximum Scaler for regression targets.

        Args:
            minimum: Minimum value of all target values.
            maximum: Maximum value of all target values.

        Raises:
            ValueError: If minimum is not less than maximum.
        """
        if not minimum < maximum:
            raise ValueError(
                f"Minimum '{minimum}' must be less than maximum '{maximum}'."
            )
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def encode(self, x: float) -> float:
        """Encode a target value by scaling it between the minimum and maximum.

        Args:
            x: Target value.

        Returns:
            Scaled target value.
        """
        return (x - self.minimum) / (self.maximum - self.minimum)

    def decode(self, x: float) -> float:
        """Decode a target value by reversing the scaling between the minimum
        and maximum. Inverse operation of encode.

        Args:
            x: Scaled target value.

        Returns:
            Unscaled target value.
        """
        return x * (self.maximum - self.minimum) + self.minimum

    def predict_batch(self, x: torch.Tensor) -> Union[List[float], float]:
        """Get encoded predictions from a batch of model outputs by
        squeezing the tensor and converting it to a list.


        Args:
            x: Batch of model outputs.

        Returns:
            Encoded predictions.
        """
        return x.squeeze().tolist()

    def majority_vote(self, x: List[float]) -> float:
        """Get the majority vote from a list of target values by averaging
        the predictions.

        Args:
            x: List of target values.

        Returns:
            Average target value.
        """
        return sum(x) / len(x)

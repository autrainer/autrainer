from typing import Dict, List, Union

import torch

from .abstract_target_transform import AbstractTargetTransform


class MinMaxScaler(AbstractTargetTransform):
    def __init__(self, target: str, minimum: float, maximum: float) -> None:
        """Minimum-Maximum Scaler for regression targets.

        Args:
            target: Name of the target.
            minimum: Minimum value of all target values.
            maximum: Maximum value of all target values.

        Raises:
            ValueError: If minimum is not less than maximum.
        """
        if not minimum < maximum:
            raise ValueError(
                f"Minimum '{minimum}' must be less than maximum '{maximum}'."
            )
        self.target = target
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def __len__(self) -> int:
        """Get the number of unique targets.

        Returns:
            Number of unique targets.
        """
        return 1

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

    def probabilities_training(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded probabilities from a batch of model outputs
        during training by applying the sigmoid function.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded probabilities.
        """
        return torch.sigmoid(x)

    def probabilities_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded probabilities from a batch of model outputs by
        applying the sigmoid function.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded probabilities.
        """
        return torch.sigmoid(x)

    def predict_inference(self, x: torch.Tensor) -> Union[List[float], float]:
        """Get the encoded predictions from a batch of model output
        probabilities by returning the raw values.

        Args:
            x: Batch of model output probabilities.

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

    def probabilities_to_dict(self, x: torch.Tensor) -> Dict[str, float]:
        """Convert a tensor of probabilities to a dictionary of targets and
        their probabilities.

        Args:
            x: Tensor of probabilities.

        Returns:
            Dictionary of targets and their probabilities.
        """
        return {self.target: x.item()}

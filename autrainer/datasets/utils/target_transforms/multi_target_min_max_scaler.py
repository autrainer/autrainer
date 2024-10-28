from typing import Dict, List

import torch

from .abstract_target_transform import AbstractTargetTransform


class MultiTargetMinMaxScaler(AbstractTargetTransform):
    def __init__(
        self,
        target: List[str],
        minimum: List[float],
        maximum: List[float],
    ) -> None:
        """Minimum-Maximum Scaler for multi-target regression.

        Args:
            target: Names of the targets.
            minimum: Minimum values of all target values.
            maximum: Maximum values of all target values.

        Raises:
            ValueError: If minimum is not less than maximum.
        """
        for m, M in zip(minimum, maximum):
            if not m < M:
                raise ValueError(
                    f"Minimum '{m}' must be less than maximum '{M}'."
                )
        self.target = target
        self.minimum = [float(m) for m in minimum]
        self._m = torch.Tensor(self.minimum)
        self.maximum = [float(M) for M in maximum]
        self._M = torch.Tensor(self.maximum)

    def __len__(self) -> int:
        """Get the number of unique targets.

        Returns:
            Number of unique targets.
        """
        return len(self.target)

    def encode(self, x: List[float]) -> torch.Tensor:
        """Encode a target value by scaling it between the minimum and maximum.

        Args:
            x: Target value.

        Returns:
            Scaled target value.
        """
        return (torch.Tensor(x) - self._m) / (self._M - self._m)

    def decode(self, x: List[float]) -> List[float]:
        """Decode a target value by reversing the scaling between the minimum
        and maximum. Inverse operation of encode.

        Args:
            x: Scaled target value.

        Returns:
            Unscaled target value.
        """
        return (torch.Tensor(x) * (self._M - self._m) + self._m).tolist()

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

    def predict_inference(self, x: torch.Tensor) -> List[List[float]]:
        """Get the encoded predictions from a batch of model output
        probabilities by returning the raw values.

        Args:
            x: Batch of model output probabilities.

        Returns:
            Encoded predictions.
        """
        return x.tolist()

    def majority_vote(self, x: List[List[float]]) -> List[float]:
        """Get the majority vote from a list of target values by averaging
        the predictions.

        Args:
            x: List of target values.

        Returns:
            Average target value.
        """
        return [sum(item) / len(item) for item in zip(*x)]

    def probabilities_to_dict(self, x: torch.Tensor) -> Dict[str, float]:
        """Convert a tensor of probabilities to a dictionary of targets and
        their probabilities.

        Args:
            x: Tensor of probabilities.

        Returns:
            Dictionary of targets and their probabilities.
        """
        return {label: prob.item() for label, prob in zip(self.target, x)}

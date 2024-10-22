from typing import List, Union

import numpy as np
import torch

from .abstract_target_transform import AbstractTargetTransform


class MultiLabelEncoder(AbstractTargetTransform):
    def __init__(self, threshold: float, labels: List[str]) -> None:
        """Multi-label encoder for multi-label classification targets.

        Args:
            threshold: Class-wise prediction threshold.
            labels: List of target labels.

        Raises:
            ValueError: If threshold is not between 0 and 1.
        """
        if not 0 <= threshold <= 1:
            raise ValueError(
                f"Threshold '{threshold}' must be between 0 and 1."
            )
        self.threshold = threshold
        self.labels = labels

    def encode(self, x: Union[List[int], List[str]]) -> torch.Tensor:
        """Encode a list of target labels by creating a binary tensor where
        each element is 1 if the label is in the list of target labels and 0
        otherwise.

        If the input is already a list of integers, it is not encoded.

        Args:
            x: List of target labels or list of integers.

        Returns:
            Binary tensor of encoded target labels.
        """
        if len(x) == 0:
            return torch.zeros(len(self.labels))
        if isinstance(x, (torch.Tensor, np.ndarray)) or all(
            isinstance(i, (int, float)) for i in x
        ):
            return torch.Tensor(x)
        return torch.Tensor([label in x for label in self.labels])

    def decode(self, x: List[int]) -> List[str]:
        """Decode a binary tensor of encoded target labels to a list of target
        labels.

        Args:
            x: Binary tensor of encoded target labels.

        Returns:
            List of target labels.
        """
        return [label for i, label in enumerate(self.labels) if x[i]]

    def probabilities_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded probabilities from a batch of model outputs by
        applying the sigmoid function.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded probabilities.
        """
        return torch.sigmoid(x)

    def predict_batch(
        self, x: torch.Tensor
    ) -> Union[List[List[int]], List[int]]:
        """Get the encoded predictions from a batch of model output
        probabilities by thresholding the probabilities.

        Args:
            x: Batch of model output probabilities.

        Returns:
            Binary tensor of encoded predictions.
        """
        return (x > self.threshold).int().squeeze().tolist()

    def majority_vote(self, x: List[List[str]]) -> List[str]:
        """Get the majority vote from a list of lists of decoded target labels
        for each label. If a label is predicted by at least half of the
        predictions, it is included in the majority vote.

        Args:
            x: List of lists of decoded target labels.

        Returns:
            List of target labels in the majority vote.
        """
        x = np.array([self.encode(i).tolist() for i in x])
        x = np.mean(x, axis=0).round().astype(int).tolist()
        return self.decode(x)

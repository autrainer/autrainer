from abc import ABC, abstractmethod
from typing import Any, List, Union

import audobject
import numpy as np
import torch


class AbstractTargetTransform(ABC, audobject.Object):
    def __init__(self) -> None:
        """Abstract target transform for handling target or label
        transformations in a dataset.

        Serves as the base for creating custom target transforms that handle
        encoding and decoding targets or labels of a dataset, obtaining
        predictions from a batch of model outputs, and determining the majority
        vote from a list of targets.
        """

    def __call__(self, x: Any) -> Any:
        return self.encode(x)

    @abstractmethod
    def encode(self, x: Any) -> Any:
        """Encode a target or label.

        Args:
            x: Target or label.

        Returns:
            Encoded target or label.
        """

    @abstractmethod
    def decode(self, x: Any) -> Any:
        """Decode a target or label. Serve as the inverse operation of encode.

        Args:
            x: Encoded target or label.

        Returns:
            Decoded target or label.
        """

    @abstractmethod
    def predict_batch(self, x: torch.Tensor) -> Union[List[Any], Any]:
        """Get encoded predictions from a batch of model outputs.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded predictions.
        """

    @abstractmethod
    def majority_vote(self, x: List[Any]) -> Any:
        """Get the majority vote from a list of decoded targets or labels.

        The majority vote is defined by the subclasses and may be the most
        frequent target, the average of all targets, or any other operation
        that determines a majority vote based on the list of targets or labels.

        Args:
            x: List of decoded targets or labels.

        Returns:
            Decoded majority vote.
        """


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

    def predict_batch(
        self, x: torch.Tensor
    ) -> Union[List[List[int]], List[int]]:
        """Predict the binary tensor of encoded predictions by thresholding the
        model outputs.

        Args:
            x: Batch of model outputs.

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


class LabelEncoder(AbstractTargetTransform):
    def __init__(self, labels: List[str]) -> None:
        """Label encoder for single-label classification targets.

        Args:
            labels: List of target labels.
        """
        self.labels = sorted(labels)
        codes = range(len(self.labels))
        self.inverse_map = {
            code: label for code, label in zip(codes, self.labels)
        }
        self.map = {label: code for code, label in zip(codes, self.labels)}

    def encode(self, x: str) -> int:
        """Encode a target label by mapping it to an integer.

        Args:
            x: Target label.

        Returns:
            Encoded target label.
        """
        return self.map[x]

    def decode(self, x: int) -> str:
        """Decode an encoded target label integer by mapping it to a label.

        Args:
            x: Encoded target label.

        Returns:
            Decoded target label.
        """
        return self.inverse_map[x]

    def predict_batch(self, x: torch.Tensor) -> Union[List[int], int]:
        """Get the encoded predictions from a batch of model outputs by
        obtaining the index of the maximum value.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded predictions.
        """
        return x.argmax(-1).squeeze().tolist()

    def majority_vote(self, x: List[str]) -> str:
        """Get the majority vote from a list of decoded labels by
        determining the most frequently predicted label.

        Args:
            x: List of decoded labels.

        Returns:
            Decoded majority vote.
        """
        return max(set(x), key=x.count)

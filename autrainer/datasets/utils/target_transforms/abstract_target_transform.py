from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import audobject
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
    def __len__(self) -> int:
        """Get the number of unique targets or labels.

        Returns:
            Number of unique targets or labels.
        """

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
    def probabilities_training(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded probabilities from a batch of model outputs
        during training.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded probabilities.
        """

    @abstractmethod
    def probabilities_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded probabilities from a batch of model outputs.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded probabilities.
        """

    @abstractmethod
    def predict_inference(self, x: torch.Tensor) -> Union[List[Any], Any]:
        """Get the encoded predictions from a batch of model output
        probabilities.

        Args:
            x: Batch of model output probabilities.

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

    @abstractmethod
    def probabilities_to_dict(self, x: torch.Tensor) -> Dict[str, float]:
        """Convert a tensor of probabilities to a dictionary of targets or
        labels and their probabilities.

        Args:
            x: Tensor of probabilities.

        Returns:
            Dictionary of targets or labels and their probabilities.
        """

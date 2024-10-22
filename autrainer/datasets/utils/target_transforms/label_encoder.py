from typing import List, Union

import torch

from .abstract_target_transform import AbstractTargetTransform


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

    def probabilities_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded probabilities from a batch of model outputs by
        applying the softmax function.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded probabilities.
        """
        return torch.softmax(x, dim=-1)

    def predict_batch(self, x: torch.Tensor) -> Union[List[int], int]:
        """Get the encoded predictions from a batch of model output
        probabilities by obtaining the index of the maximum value.

        Args:
            x: Batch of model output probabilities.

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
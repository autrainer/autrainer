from typing import Dict, List, Union

import numpy as np
import torch

from .abstract_target_transform import AbstractTargetTransform


class SEDEncoder(AbstractTargetTransform):
    def __init__(
        self,
        labels: List[str],
        frame_rate: float = 0.08,
        duration: float = 10,
        threshold: float = 0.5,
        min_event_length: float = 0.3,
        pause_length: float = 0.5,
    ) -> None:
        """Label encoder for multi-label SED targets.

        Args:
            labels: List of target labels.
        """
        self.labels = sorted(labels)
        codes = range(len(self.labels))
        self.inverse_map = {
            code: label for code, label in zip(codes, self.labels)
        }
        self.map = {label: code for code, label in zip(codes, self.labels)}
        self.frame_rate = frame_rate
        self.duration = duration
        self.min_event_length = min_event_length
        self.pause_length = pause_length
        if not 0 <= threshold <= 1:
            raise ValueError(
                f"Threshold '{threshold}' must be between 0 and 1."
            )
        self.threshold = threshold

    def __len__(self) -> int:
        """Get the number of unique target labels.

        Returns:
            Number of unique target labels.
        """
        return len(self.labels)

    def encode(self, x: str) -> int:
        """Encode a target label by mapping it to an integer.

        Args:
            x: Target label.

        Returns:
            Encoded target label.
        """
        return self.map[x]

    def decode(self, x: np.ndarray) -> Dict:
        """Decode an encoded target label integer by mapping it to a label.

        Args:
            x: Encoded target label.

        Returns:
            Decoded target label.
        """
        res = []
        for col, label in enumerate(self.labels):
            pred_vector = x[:, col]
            # look for transitions from 0<->1
            diffs = np.diff(pred_vector, 1)
            # onsets defined as 0->1
            onsets = list(np.where(diffs == 1)[0])
            # offsets defined as 1->0
            offsets = list(np.where(diffs == -1)[0])
            if pred_vector[0] == 1:
                # check if event at start of file
                onsets = [0] + onsets
            if pred_vector[-1] == 1:
                # check if event at end of file
                offsets = offsets + [len(pred_vector)]
            # TODO: fuse events with short pauses
            for on, off in zip(onsets, offsets):
                if (off - on) * self.frame_rate > self.min_event_length:
                    # discard short events
                    res.append(
                        {
                            "event_label": label,
                            "event_onset": on * self.frame_rate,
                            "event_offset": off * self.frame_rate,
                        }
                    )
        return res

    def probabilities_training(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded probabilities from a batch of model outputs
        during training by returning the raw model outputs.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded probabilities.
        """
        return x

    def probabilities_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded probabilities from a batch of model outputs by
        applying the sigmoid function.

        Args:
            x: Batch of model outputs.

        Returns:
            Encoded probabilities.
        """
        return torch.sigmoid(x)

    def predict_inference(self, x: torch.Tensor) -> Union[List[int], int]:
        """Get the encoded predictions from a batch of model output
        probabilities by obtaining the index of the maximum value.

        Args:
            x: Batch of model output probabilities.

        Returns:
            Encoded predictions.
        """
        return (x > self.threshold).int().squeeze()

    def majority_vote(self, x: List[str]) -> str:
        """Get the majority vote from a list of decoded labels by
        determining the event with the longest duration.

        Args:
            x: List of decoded labels.

        Returns:
            Decoded majority vote.
        """
        return max(x, key=x["event_offset"] - x["event_onset"])

    def probabilities_to_dict(self, x: torch.Tensor) -> Dict[str, float]:
        """Convert a tensor of probabilities to a dictionary of labels and
        their probabilities.

        Args:
            x: Tensor of probabilities.

        Returns:
            Dictionary of labels and their probabilities.
        """
        return {self.decode(i): prob.item() for i, prob in enumerate(x)}

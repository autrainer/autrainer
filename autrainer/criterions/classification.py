from typing import TYPE_CHECKING, Dict, List

import pandas as pd
import torch


if TYPE_CHECKING:  # pragma: no cover
    from autrainer.datasets import AbstractDataset


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.CrossEntropyLoss.forward`.

        Converts the targets to `long` if it is a 1D tensor.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        if y.ndim == 1:
            y = y.long()
        return super().forward(x, y)


class BalancedCrossEntropyLoss(CrossEntropyLoss):
    def setup(self, data: "AbstractDataset") -> None:
        """Calculate balanced weights for the dataset based on the target
        frequency in the training set.

        Args:
            data: Instance of the dataset.
        """
        frequency = (
            data.df_train[data.target_column]
            .map(data.target_transform)
            .value_counts()
            .sort_index()
            .values
        )
        weight = torch.tensor(1 / frequency, dtype=torch.float32)
        self.weight = weight * len(weight) / weight.sum()


class WeightedCrossEntropyLoss(BalancedCrossEntropyLoss):
    def __init__(self, class_weights: Dict[str, float], **kwargs) -> None:
        """Wrapper for `torch.nn.CrossEntropyLoss` with manual class weights.

        The class weights are automatically normalized to sum up to the number
        of classes.

        Args:
            class_weights: Dictionary with class weights corresponding to the
                target labels and their respective weights.
            **kwargs: Additional keyword arguments passed to
                `torch.nn.CrossEntropyLoss`.
        """
        self.class_weights = class_weights
        super().__init__(**kwargs)

    def setup(self, data: "AbstractDataset") -> None:
        """Calculate the class weights based on the provided dictionary.

        Args:
            data: Instance of the dataset.
        """
        values = []

        for label in data.target_transform.labels:
            if label not in self.class_weights:
                raise ValueError(f"Missing class weight for label '{label}'.")
            values.append(self.class_weights[label])

        weight = torch.tensor(values, dtype=torch.float32)
        self.weight = weight * len(weight) / weight.sum()


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.BCEWithLogitsLoss.forward`.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        return super().forward(x, y.float())


class BalancedBCEWithLogitsLoss(BCEWithLogitsLoss):
    weight: torch.Tensor

    def setup(self, data: "AbstractDataset") -> None:
        """Calculate balanced weights for the dataset based on the target
        frequency in the training set.

        Args:
            data: Instance of the dataset.
        """

        def encode(x: pd.Series) -> List[int]:
            return data.target_transform(x.to_list()).tolist()

        frequency = (
            pd.DataFrame(
                data.df_train[data.target_column]
                .apply(encode, axis=1)
                .to_list(),
                columns=data.target_transform.labels,
            )
            .sum(axis=0)
            .values
        )
        weight = torch.tensor(1 / frequency, dtype=torch.float32)
        weight = weight * len(weight) / weight.sum()
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.BCEWithLogitsLoss.forward` with balanced
        weights.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        return super().forward(x, y) * self.weight.expand_as(y)


class WeightedBCEWithLogitsLoss(BalancedBCEWithLogitsLoss):
    def __init__(self, class_weights: Dict[str, float], **kwargs) -> None:
        """Wrapper for `torch.nn.BCEWithLogitsLoss` with manual class weights.

        The class weights are automatically normalized to sum up to the number
        of classes.

        Args:
            class_weights: Dictionary with class weights corresponding to the
                target labels and their respective weights.
            **kwargs: Additional keyword arguments passed to
                `torch.nn.BCEWithLogitsLoss`.
        """
        self.class_weights = class_weights
        super().__init__(**kwargs)

    def setup(self, data: "AbstractDataset") -> None:
        """Calculate the class weights based on the provided dictionary.

        Args:
            data: Instance of the dataset.
        """
        values = []

        for label in data.target_transform.labels:
            if label not in self.class_weights:
                raise ValueError(f"Missing class weight for label '{label}'.")
            values.append(self.class_weights[label])

        weight = torch.tensor(values, dtype=torch.float32)
        weight = weight * len(weight) / weight.sum()
        self.register_buffer("weight", weight)
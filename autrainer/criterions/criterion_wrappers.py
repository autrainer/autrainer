from typing import TYPE_CHECKING, List

import pandas as pd
import torch


if TYPE_CHECKING:
    from autrainer.datasets import AbstractDataset  # pragma: no cover


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


class BalancedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
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
        return super().forward(x, y.float()) * self.weight.expand_as(y)


class MSELoss(torch.nn.MSELoss):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Wrapper for `torch.nn.MSELoss.forward`.

        Squeezes the model outputs along the last dimension to match the shape
        of the targets.
        Converts the targets to `float`.

        Args:
            x: Batched model outputs.
            y: Targets.

        Returns:
            Loss.
        """
        # TODO: is there a more elegant way to handle broadcasting
        return super().forward(x.squeeze(-1), y.float())

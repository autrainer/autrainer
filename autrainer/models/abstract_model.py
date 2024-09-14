from abc import ABC, abstractmethod

import audobject
import torch


class AbstractModel(torch.nn.Module, audobject.Object, ABC):
    def __init__(self, output_dim: int) -> None:
        """Abstract model class.

        Args:
            output_dim: Output dimension of the model.
        """
        super().__init__()
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """

    @abstractmethod
    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from the model.

        Args:
            x: Input tensor.

        Returns:
            Embeddings.
        """

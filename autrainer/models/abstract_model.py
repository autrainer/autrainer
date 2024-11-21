from abc import ABC, abstractmethod

import audobject
import torch

from autrainer.datasets.utils.data_struct import Data


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


class BaseModelWrapper:
    def __init__(self, model: AbstractModel):
        self.model = model

    @staticmethod
    def unwrap_data(data: Data):
        return data.features

    def __call__(self, data: Data):
        return self.model(self.unwrap_data(data))

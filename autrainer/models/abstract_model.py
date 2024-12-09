from abc import ABC, abstractmethod
from typing import Union

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

    def _parse_data(self, x: Union[Data, torch.Tensor]) -> torch.Tensor:
        """Parsing input data.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        if isinstance(x, Data):
            return x.features
        else:
            return x

    @abstractmethod
    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from the model.

        Args:
            x: Input tensor.

        Returns:
            Embeddings.
        """

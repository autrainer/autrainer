from abc import ABC, abstractmethod
from functools import cached_property
from inspect import signature
from typing import List

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
    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """Get embeddings from the model.

        Args:
            features: Input tensor.

        Returns:
            Embeddings.
        """

    @cached_property
    def inputs(self) -> List[str]:
        """Get the inputs to the model's forward method.

        Returns:
            Model inputs.
        """
        names = [v.name for v in signature(self.forward).parameters.values()]
        if names[0] != "features":
            raise NameError(
                (
                    f"Model {type(self).__name__} "
                    "does not have 'features' "
                    "as the first argument of its 'forward' method. "
                    f"Its arguments are: {names}. "
                    "Please rewrite the 'forward' method accordingly."
                )
            )
        return names

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Get model output.

        Args:
            features: Input tensor.

        Returns:
            Model output.
        """
        raise NotImplementedError

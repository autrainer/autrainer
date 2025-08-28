from abc import ABC, abstractmethod
from functools import cached_property
from inspect import signature
from typing import Callable, List, Optional, Union

import audobject
import torch


class AbstractModel(torch.nn.Module, audobject.Object, ABC):
    def __init__(
        self,
        output_dim: int,
        transfer: Optional[Union[bool, str]] = None,
    ) -> None:
        """Abstract model class.

        Args:
            output_dim: Output dimension of the model.
            transfer: Whether to load the model with pretrained weights if
                available. May be a boolean or a string representing a truthy
                or falsy value. Defaults to None.
        """
        super().__init__()
        self.output_dim = output_dim
        self.transfer = transfer
        self.inputs  # precompute and verify inputs  # noqa: B018
        self.embedding_inputs  # precompute and verify embedding inputs  # noqa: B018

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
        return self._collect_inputs(self.forward)

    @cached_property
    def embedding_inputs(self) -> List[str]:
        """Get the inputs to the model's embedding method.

        Returns:
            Model inputs.
        """
        return self._collect_inputs(self.embeddings)

    def _collect_inputs(self, fn: Callable) -> List[str]:
        names = [v.name for v in signature(fn).parameters.values()]
        fn_name = fn.__name__
        if names[0] != "features":
            raise NameError(
                (
                    f"Model {type(self).__name__} "
                    "does not have 'features' "
                    f"as the first argument of its '{fn_name}' method. "
                    f"Its arguments are: {names}. "
                    f"Please rewrite the '{fn_name}' method accordingly."
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
        raise NotImplementedError  # pragma: no cover

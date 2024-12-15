from typing import List, Tuple

import torch
import torchvision

from .abstract_model import AbstractModel
from .utils import ExtractLayerEmbeddings


class TorchvisionModel(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        torchvision_name: str,
        transfer=False,
        **kwargs,
    ) -> None:
        """Wrapper for torchvision models.

        Args:
            output_dim: Number of output classes.
            torchvision_name: Name of the model available in
                `torchvision.models`.
            transfer: Whether to load the model with pretrained weights.
                The "DEFAULT" weights are used if `transfer` is True.
                The final layer is replaced with a new layer with `output_dim`
                output features. Defaults to False.
            kwargs: Additional arguments to pass to the model constructor.
        """
        super().__init__(output_dim)
        self.torchvision_name = torchvision_name
        self.transfer = transfer
        for key, value in kwargs.items():
            setattr(self, key, value)  # pragma: no cover

        if transfer:
            self.model = torchvision.models.get_model(
                torchvision_name,
                weights="DEFAULT",
                **kwargs,
            )
            self._replace_final_layer(self.model)
        else:
            self.model = torchvision.models.get_model(
                torchvision_name,
                weights=None,
                num_classes=output_dim,
                **kwargs,
            )
        self._embedding_extractor = ExtractLayerEmbeddings(self.model)

    def _replace_final_layer(self, module: torch.nn.Module) -> None:
        """Replace the final linear layer of the model with a new linear
        layer having `self.output_dim` output features.

        Args:
            module: Model to replace the final layer.

        Raises:
            ValueError: If no linear layer is found in the model.
        """
        layers: List[Tuple[torch.nn.Module, str, torch.nn.Linear]] = []

        def collect_layers(module: torch.nn.Module):
            for name, layer in module.named_children():
                if isinstance(layer, torch.nn.Linear):
                    layers.append((module, name, layer))
                else:
                    collect_layers(layer)

        collect_layers(module)
        if layers:
            module, name, layer = layers[-1]
            in_features = layer.in_features
            new_layer = torch.nn.Linear(in_features, self.output_dim)
            setattr(module, name, new_layer)
        else:
            raise ValueError(
                f"Failed to override last linear layer for model "
                f"'{self.torchvision_name}'."
            )  # pragma: no cover

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

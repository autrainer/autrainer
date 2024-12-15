import logging

import timm
import torch

from .abstract_model import AbstractModel
from .utils import ExtractLayerEmbeddings


class TimmModel(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        timm_name: str,
        transfer=False,
        **kwargs,
    ) -> None:
        """Wrapper for timm models.

        Args:
            output_dim: Number of output classes.
            timm_name: Name of the model available in `timm.create_model`.
            transfer: Whether to load the model with pretrained weights.
                The final layer is replaced with a new layer with `output_dim`
                output features. Defaults to False.
            kwargs: Additional arguments to pass to the model constructor.
        """
        super().__init__(output_dim)
        self.timm_name = timm_name
        self.transfer = transfer
        for key, value in kwargs.items():
            setattr(self, key, value)

        logging.getLogger("timm").setLevel(logging.ERROR)
        self.model = timm.create_model(
            timm_name,
            pretrained=transfer,
            num_classes=output_dim,
            **kwargs,
        )
        self._embedding_extractor = ExtractLayerEmbeddings(self.model)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

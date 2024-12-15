import warnings

import torch
from transformers import Wav2Vec2Model

from .abstract_model import AbstractModel
from .ffnn import FFNN


class W2V2Backbone(AbstractModel):
    def __init__(
        self,
        model_name,
        freeze_extractor: bool = True,
        time_pooling: bool = True,
    ) -> None:
        self.model_name = model_name
        self.freeze_extractor = freeze_extractor
        self.time_pooling = time_pooling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Wav2Vec2Model.from_pretrained(self.model_name)
        super().__init__(output_dim=model.config.hidden_size)
        self.model = model
        if self.freeze_extractor:
            self.model.freeze_feature_encoder()

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)["last_hidden_state"]
        if self.time_pooling:
            x = x.mean(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x)


class W2V2FFNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        model_name: str,
        freeze_extractor: bool,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        """Wav2Vec2 model with FFNN frontend adapted for audio classification.
        For more information, see:
        https://huggingface.co/docs/transformers/model_doc/wav2vec2

        Args:
            output_dim: Output dimension of the FFNN.
            model_name: Name of the model loaded from Huggingface.
            freeze_extractor: Whether to freeze the feature extractor.
            hidden_size: Hidden size of the FFNN.
            num_layers: Number of layers of the FFNN. Defaults to 2.
            dropout: Dropout rate. Defaults to 0.5.
        """
        super().__init__(output_dim)
        self.model_name = model_name
        self.freeze_extractor = freeze_extractor
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone = W2V2Backbone(
            model_name=model_name,
            freeze_extractor=freeze_extractor,
            time_pooling=True,
        )
        self.frontend = FFNN(
            input_size=self.backbone.output_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x.squeeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frontend(self.embeddings(x))

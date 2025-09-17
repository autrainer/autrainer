import warnings

import torch
from transformers import WhisperConfig, WhisperModel

from .abstract_model import AbstractModel
from .ffnn import FFNN


class WhisperBackbone(AbstractModel):
    def __init__(self, model_name: str, transfer: bool = False) -> None:
        self.model_name = model_name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if transfer:  # pragma: no cover
                model = WhisperModel.from_pretrained(self.model_name)
            else:
                config = WhisperConfig.from_pretrained(
                    self.model_name,
                    output_hidden_states=True,
                    return_dict=True,
                )
                model = WhisperModel(config)
        super().__init__(
            output_dim=model.config.hidden_size,
            transfer=transfer,
        )
        self.model = model
        self.register_buffer(
            "decoder_input_ids",
            torch.tensor([1, 1]) * self.model.config.decoder_start_token_id,
        )

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        ids = torch.stack([self.decoder_input_ids] * features.shape[0])
        return self.model(features, decoder_input_ids=ids)["last_hidden_state"][:, 0, :]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.embeddings(features)


class WhisperFFNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        model_name: str,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        transfer: bool = False,
    ) -> None:
        """Whisper model with FFNN frontend adapted for audio classification.
        For more information, see:
        https://doi.org/10.48550/arXiv.2212.04356

        Args:
            model_name: Name of the model loaded from Huggingface.
            hidden_size: Hidden size of the FFNN.
            output_dim: Output dimension of the FFNN.
            num_layers: Number of layers of the FFNN. Defaults to 2.
            dropout: Dropout rate. Defaults to 0.5.
            transfer: Whether to initialize the Whisper backbone with
                pretrained weights. Defaults to False.
        """
        super().__init__(output_dim, transfer)
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone = WhisperBackbone(model_name, transfer)
        self.frontend = FFNN(
            input_size=self.backbone.output_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.frontend(self.embeddings(features))

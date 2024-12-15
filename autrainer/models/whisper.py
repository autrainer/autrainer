import torch
from transformers import WhisperModel

from .abstract_model import AbstractModel
from .ffnn import FFNN


class WhisperBackbone(AbstractModel):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        model = WhisperModel.from_pretrained(self.model_name)
        super().__init__(output_dim=model.config.hidden_size)
        self.model = model
        self.register_buffer(
            "decoder_input_ids",
            torch.tensor([1, 1]) * self.model.config.decoder_start_token_id,
        )

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        decoder_input_ids = torch.stack([self.decoder_input_ids] * x.shape[0])
        x = self.model(x, decoder_input_ids=decoder_input_ids)[
            "last_hidden_state"
        ][:, 0, :]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x)


class WhisperFFNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        model_name: str,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
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
        """
        super().__init__(output_dim)
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone = WhisperBackbone(model_name=model_name)
        self.frontend = FFNN(
            input_size=self.backbone.output_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frontend(self.embeddings(x))

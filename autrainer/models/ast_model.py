from typing import Optional

import torch
from transformers import ASTConfig
from transformers import ASTModel as ASTBaseModel

from .abstract_model import AbstractModel


class ASTModel(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        num_hidden_layers: int = 12,
        hidden_size: int = 128,
        dropout: float = 0.5,
        transfer: Optional[str] = None,
    ) -> None:
        """Audio Speech Transformer (AST) model. For more information see:
        https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/audio-spectrogram-transformer

        Args:
            output_dim: Output dimension of the model.
            num_hidden_layers: Number of hidden layers in the transformer.
                Defaults to 12.
            hidden_size: Hidden size of the linear layer. Defaults to 128.
            dropout: Dropout rate. Defaults to 0.5.
            transfer: Name of the pretrained model to load. If None, the default
                AST fine-tuned on AudioSet is used. Defaults to None.
                For more information see:
                https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
        """
        super().__init__(output_dim)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.transfer = transfer
        if self.transfer is not None:
            self.model = ASTBaseModel.from_pretrained(
                self.transfer,
                num_hidden_layers=num_hidden_layers,
            )
        else:
            config = ASTConfig(
                num_hidden_layers=num_hidden_layers,
            )
            self.model = ASTBaseModel(config)
        layers = [
            torch.nn.Linear(self.model.config.hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, output_dim),
        ]
        self.out = torch.nn.Sequential(*layers)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).last_hidden_state.mean(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.out(x)
        return x

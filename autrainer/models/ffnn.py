import torch

from .abstract_model import AbstractModel
from .utils import ExtractLayerEmbeddings


class FFNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        """Feedforward neural network.

        Args:
            output_dim: Output dimension.
            input_size: Input size.
            hidden_size: Hidden size.
            num_layers: Number of layers.
            dropout: Dropout rate.
        """
        super().__init__(output_dim)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []
        layer_input = input_size
        for i in range(num_layers - 1):
            layers += [
                (f"Linear{i}", torch.nn.Linear(layer_input, hidden_size)),
                (f"ReLU{i}", torch.nn.ReLU()),
                (f"Dropout{i}", torch.nn.Dropout(dropout)),
            ]
            layer_input = hidden_size
        layers.append(
            (
                f"Linear{num_layers - 1}",
                torch.nn.Linear(layer_input, output_dim),
            )
        )

        for name, layer in layers:
            self.add_module(name, layer)

        self._embedding_extractor = ExtractLayerEmbeddings(self)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.children():
            x = layer(x)
        return x

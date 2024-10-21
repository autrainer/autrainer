import torch

from .abstract_model import AbstractModel
from .ffnn import FFNN


class Sequential(AbstractModel):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.5,
        cell: str = "LSTM",
        time_pooling: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__(
            output_dim=2 * hidden_size if bidirectional else hidden_size
        )
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.time_pooling = time_pooling
        self.bidirectional = bidirectional
        self.cell = cell
        if self.cell not in ["LSTM", "GRU"]:
            raise NotImplementedError(self.cell)
        cell_cls = torch.nn.LSTM if self.cell == "LSTM" else torch.nn.GRU
        self.model = cell_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)[0]
        if self.time_pooling:
            x = x.mean(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x)


class SeqFFNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        backbone_input_dim: int,
        backbone_hidden_size: int,
        backbone_num_layers: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        backbone_dropout: float = 0.5,
        backbone_cell: str = "LSTM",
        backbone_time_pooling: bool = True,
        backbone_bidirectional: bool = False,
    ):
        """Sequential model with FFNN frontend.

        Args:
            output_dim: Output dimension of the FFNN.
            backbone_input_dim: Input dimension of the backbone.
            backbone_hidden_size: Hidden size of the backbone.
            backbone_num_layers: Number of layers of the backbone.
            hidden_size: Hidden size of the FFNN.
            num_layers: Number of layers of the FFNN. Defaults to 2.
            dropout: Dropout rate of the FFNN. Defaults to 0.5.
            backbone_dropout: Dropout rate of the backbone. Defaults to 0.5.
            backbone_cell: Cell type of the backbone in ["LSTM", "GRU"].
                Defaults to "LSTM".
            backbone_time_pooling: Whether to apply time pooling in the
                backbone. Defaults to True.
            backbone_bidirectional: Whether to use a bidirectional backbone.
                Defaults to False.
        """
        super().__init__(output_dim)
        self.backbone_input_dim = backbone_input_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.backbone_num_layers = backbone_num_layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone_dropout = backbone_dropout
        self.backbone_cell = backbone_cell
        self.backbone_time_pooling = backbone_time_pooling
        self.backbone_bidirectional = backbone_bidirectional
        self.backbone = Sequential(
            input_dim=backbone_input_dim,
            hidden_size=backbone_hidden_size,
            num_layers=backbone_num_layers,
            dropout=backbone_dropout,
            cell=backbone_cell,
            time_pooling=backbone_time_pooling,
            bidirectional=backbone_bidirectional,
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

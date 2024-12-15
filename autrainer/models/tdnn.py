import os
import warnings

from speechbrain.inference.classifiers import EncoderClassifier
import torch

from .abstract_model import AbstractModel
from .ffnn import FFNN


class TDNNFFNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        """Time Delay Neural Network with FFNN frontend.

        Args:
            output_dim: Output dimension.
            hidden_size: Hidden size.
            num_layers: Number of layers. Defaults to 2.
            dropout: Dropout rate. Defaults to 0.5.
        """
        super().__init__(output_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        checkpoint_dir = os.path.join(torch.hub.get_dir(), "speechbrain")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            tdnn = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                freeze_params=False,
                savedir=checkpoint_dir,
            )
        self.backbone = tdnn.mods["embedding_model"]
        self.features = tdnn.mods["compute_features"]
        self.frontend = FFNN(
            input_size=192,  # TODO: get from model?
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        _device = x.device
        feats = self.features(x.squeeze(1).cpu())
        feats = feats.to(_device)
        embs = self.backbone(feats).squeeze(1)
        return embs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = self.embeddings(x)
        return self.frontend(embs)

import os
import warnings

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
        transfer: bool = False,
    ) -> None:
        """Time Delay Neural Network with FFNN frontend.

        Args:
            output_dim: Output dimension.
            hidden_size: Hidden size.
            num_layers: Number of layers. Defaults to 2.
            dropout: Dropout rate. Defaults to 0.5.
            transfer: Whether to initialize the TDNN backbone with
                pretrained weights. Defaults to False.
        """
        super().__init__(output_dim, transfer)  # no transfer learning weights
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if transfer:  # pragma: no cover
                checkpoint_dir = os.path.join(torch.hub.get_dir(), "speechbrain")
                os.makedirs(checkpoint_dir, exist_ok=True)
                from speechbrain.inference.classifiers import EncoderClassifier

                tdnn = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    freeze_params=False,
                    savedir=checkpoint_dir,
                )
                self.features = tdnn.mods["compute_features"]
                self.backbone = tdnn.mods["embedding_model"]
            else:
                from speechbrain.lobes.features import Fbank
                from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

                # taken from https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/main/hyperparams.yaml
                self.features = Fbank(n_mels=80)
                self.backbone = ECAPA_TDNN(
                    input_size=80,
                    channels=[1024, 1024, 1024, 1024, 3072],
                )

        self.frontend = FFNN(
            input_size=list(self.backbone.modules())[-1].weight.shape[0],
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        _device = features.device
        features = self.features(features.squeeze(1).cpu())
        features = features.to(_device)
        return self.backbone(features).squeeze(1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embs = self.embeddings(features)
        return self.frontend(embs)

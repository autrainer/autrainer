from typing import Optional

import torch
import torch.nn.functional as F

from .abstract_model import AbstractModel
from .utils import ConvBlock, init_bn, init_layer, load_transfer_weights


class Cnn10(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        sigmoid_output: bool = False,
        sigmoid_predictions: bool = False,
        segmentwise: bool = False,
        in_channels: int = 1,
        transfer: Optional[str] = None,
    ) -> None:
        """CNN10 PANN model. For more information see:
        https://doi.org/10.48550/arXiv.1912.10211

        Args:
            output_dim: Output dimension of the model.
            sigmoid_output: Whether to apply sigmoid activation to the output.
                Defaults to False.
            sigmoid_predictions: Whether to apply sigmoid activation during
                inference. Defaults to False.
            segmentwise: Whether to use segmentwise path or clipwise path.
                Defaults to False.
            in_channels: Number of input channels. Defaults to 1.
            transfer: Link to the weights to transfer. If None, the weights
                weights will be randomly initialized. Defaults to None.
        """
        super().__init__(output_dim)
        self.sigmoid_output = sigmoid_output
        self.sigmoid_predictions = sigmoid_predictions
        self.segmentwise = segmentwise
        self.in_channels = in_channels
        self.transfer = transfer
        self.bn0 = torch.nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = torch.nn.Linear(512, 512, bias=True)
        self.out = torch.nn.Linear(512, output_dim, bias=True)

        self.init_weight()
        if self.transfer is not None:
            load_transfer_weights(self, self.transfer)

    def init_weight(self) -> None:
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x: torch.Tensor) -> torch.Tensor:
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        if (not self.training) and self.sigmoid_predictions:
            # sigmoid output for multi-label classification
            if not self.sigmoid_output:  # avoid double op
                x = torch.sigmoid(x)
        return x

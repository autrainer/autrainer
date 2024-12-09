r"""Code taken from https://github.com/end2you/end2you.

Copied only model code as needed.
Adapted to match coding styles of current repo:
- Remove input_size argument
+ Add output_dim argument
+ Changed RNN model to local Sequential
+ Added time_pooling layer before linear
  + This changes model architecture
+ Remove Emo16
  + Seems broken; 2nd max-pool applied over channels
  + Output dimension is too long
  + Model's results never reproduced anyway

"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .abstract_model import AbstractModel
from .sequential import Sequential


class Base(nn.Module):
    """Base class to build convolutional neural network model."""

    def __init__(
        self,
        conv_layers_args: dict,
        maxpool_layers_args: dict,
        conv_op: nn = nn.Conv1d,
        max_pool_op: nn = nn.MaxPool1d,
        activ_fn: nn = nn.LeakyReLU(),
        normalize: bool = False,
    ) -> None:
        """Audio model.

        Args:
            conv_layers_args: Parameters of convolutions layers.
            maxpool_layers_args: Parameters of max pool layer layers.
            conv_op: Convolution operation to use. Defaults to torch.nn.Conv1d.
            max_pool_op: Max pooling operation to use.
                Defaults to torch.nn.MaxPool1d.
            activ_fn: Activation function to use.
                Defaults to torch.nn.LeakyReLU().
            normalize: Whether to use batch normalization. Defaults to False.
        """

        super().__init__()
        self.conv_layers_args = conv_layers_args
        self.maxpool_layers_args = maxpool_layers_args
        self.conv_op = conv_op
        self.max_pool_op = max_pool_op
        self.activ_fn = activ_fn
        self.normalize = normalize

        network_layers = nn.ModuleList()
        for conv_args, mp_args in zip(
            *[conv_layers_args.values(), maxpool_layers_args.values()]
        ):
            network_layers.extend(
                [self._conv_block(conv_args, activ_fn, normalize)]
            )
            network_layers.extend([max_pool_op(**mp_args)])

        self.network = nn.Sequential(*network_layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters of the model."""
        for m in list(self.modules()):
            self._init_weights(m)

    def _init_weights(self, m: nn.Module) -> None:
        """Helper method to initialize the parameters of the model
        with Kaiming uniform initialization.

        Args:
            m: Module to initialize.
        """

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.kaiming_uniform_(param)

    @classmethod
    def _num_out_features(
        cls,
        input_size: int,
        conv_args: dict,
        mp_args: dict,
    ) -> int:
        """Number of features extracted from Convolution Neural Network.

        Args:
            input_size: Number of samples of the frame.
            conv_args: Parameters of convolutions layers.
            mp_args: Parameters of max pool layer layers.

        Returns:
            Number of features extracted from the network layers.
        """

        layer_input = input_size
        for i, (conv_arg, mp_arg) in enumerate(
            zip(*[conv_args.values(), mp_args.values()])
        ):
            # number of features in the convolution output
            layer_input = np.floor(
                (
                    layer_input
                    - conv_arg["kernel_size"]
                    + 2 * conv_arg["padding"]
                )
                / conv_arg["stride"]
                + 1
            )

            layer_input = np.floor(
                (layer_input - mp_arg["kernel_size"]) / mp_arg["stride"] + 1
            )

        return int(layer_input)

    def _conv_block(
        self,
        conv_args: dict,
        activ_fn: nn,
        normalize: bool = False,
    ) -> nn.Module:
        """Convolution block.

        Args:
            conv_args: Parameters of convolution layer.
            activ_fn: Activation function to use. Defaults to
                torch.nn.LeakyReLU().
            normalize: Whether to use batch normalization. Defaults to False.
        """

        layer = nn.ModuleList([self.conv_op(**conv_args)])

        if normalize:
            layer.append(nn.BatchNorm1d(conv_args["out_channels"]))

        layer.append(activ_fn)
        return nn.Sequential(*layer)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x)


class Emo18(nn.Module):
    def __init__(self) -> None:
        """Speech emotion recognition model proposed in:
        https://doi.org/10.1109/ICASSP.2018.8462677
        """

        super().__init__()
        self.model, self.num_features = self.build_audio_model()

    def build_audio_model(self) -> Tuple[nn.Module, int]:
        """Build the audio model: 3 blocks of convolution + max-pooling."""

        out_channels = [64, 128, 256]
        in_channels = [1]
        in_channels.extend([x for x in out_channels[:-1]])
        kernel_size = [8, 6, 6]
        stride = [1, 1, 1]
        padding = ((np.array(kernel_size) - 1) // 2).tolist()

        num_layers = len(in_channels)
        conv_args = {
            f"layer{i}": {
                "in_channels": in_channels[i],
                "out_channels": out_channels[i],
                "kernel_size": kernel_size[i],
                "stride": stride[i],
                "padding": padding[i],
            }
            for i in range(num_layers)
        }

        kernel_size = [10, 8, 8]
        stride = [10, 8, 8]
        maxpool_args = {
            f"layer{i}": {"kernel_size": kernel_size[i], "stride": stride[i]}
            for i in range(num_layers)
        }

        audio_model = Base(conv_args, maxpool_args, normalize=True)
        # conv_red_size = Base._num_out_features(
        #     input_size, conv_args, maxpool_args)
        num_layers = len(in_channels) - 1
        # num_out_features = conv_red_size * \
        #     conv_args[f"layer{num_layers}"]["out_channels"]

        num_out_features = conv_args[f"layer{num_layers}"]["out_channels"]

        return audio_model, num_out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Zhao19(nn.Module):
    def __init__(self) -> None:
        """Speech emotion recognition model proposed in:
        https://doi.org/10.1016/j.bspc.2018.08.035
        """

        super().__init__()
        self.model, self.num_features = self.build_audio_model()

    def build_audio_model(self) -> Tuple[nn.Module, int]:
        """Build the audio model: 3 blocks of convolution + max-pooling."""

        out_channels = [64, 64, 128, 128]
        in_channels = [1]
        in_channels.extend([x for x in out_channels[:-1]])
        kernel_size = [3, 3, 3, 3]
        stride = [1, 1, 1, 1]
        padding = ((np.array(kernel_size) - 1) // 2).tolist()

        num_layers = len(in_channels)
        conv_args = {
            f"layer{i}": {
                "in_channels": in_channels[i],
                "out_channels": out_channels[i],
                "kernel_size": kernel_size[i],
                "stride": stride[i],
                "padding": padding[i],
            }
            for i in range(num_layers)
        }

        kernel_size = [4, 4, 4, 4]
        stride = [4, 4, 4, 4]
        maxpool_args = {
            f"layer{i}": {"kernel_size": kernel_size[i], "stride": stride[i]}
            for i in range(num_layers)
        }

        audio_model = Base(
            conv_args, maxpool_args, normalize=True, activ_fn=nn.ELU()
        )

        return audio_model, out_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AudioModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        *args,
        **kwargs,
    ):
        """Audio network model.

        Args:
            model_name: Name of the model in ["emo18", "zhao19"].
            *args: Additional arguments to the model.
            **kwargs: Additional keyword arguments to the model.
        """

        super().__init__()

        self.model = self._get_model(model_name)
        self.model = self.model(*args, **kwargs)
        self.num_features = self.model.num_features

    def _get_model(self, model_name: str) -> nn.Module:
        """Factory method to choose audio model."""

        return {"emo18": Emo18, "zhao19": Zhao19}[model_name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AudioRNNModel(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        model_name: str,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        cell: str = "LSTM",
        bidirectional: bool = False,
    ) -> None:
        """Audio RNN model.

        Args:
            output_dim: Output dimension of the model.
            model_name: Model name in ["emo18", "zhao19"].
            hidden_size: Hidden size of the RNN. Defaults to 256.
            num_layers: Number of layers of the RNN. Defaults to 2.
            dropout: Dropout rate. Defaults to 0.5.
            cell: Type of RNN cell in ["LSTM", "GRU"] Defaults to "LSTM".
            bidirectional: Whether to use a bidirectional RNN.
                Defaults to False.
        """
        super().__init__(output_dim)
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell = cell
        self.bidirectional = bidirectional
        audio_network = AudioModel(model_name=model_name)
        self.audio_model = audio_network
        self.rnn = Sequential(
            input_dim=self.audio_model.num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell=cell,
            time_pooling=True,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(self.rnn.hidden_size, self.output_dim)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, t = x.shape
        x = x.view(batch_size * seq_length, 1, t)
        audio_out = self.audio_model(x)
        audio_out = audio_out.transpose(1, 2)

        rnn_out = self.rnn(audio_out)

        return rnn_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x ((torch.Tensor) - BS x S x 1 x T)
        """
        rnn_out = self.embeddings(x)
        output = self.linear(rnn_out)
        return output

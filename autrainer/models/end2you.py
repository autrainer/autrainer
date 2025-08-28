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

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch

from .abstract_model import AbstractModel
from .sequential import Sequential
from .utils import assert_no_transfer_weights


class Base(torch.nn.Module):
    """Base class to build convolutional neural network model."""

    def __init__(
        self,
        conv_layers_args: dict,
        maxpool_layers_args: dict,
        conv_op: Optional[Type[torch.nn.Module]] = None,
        max_pool_op: Optional[Type[torch.nn.Module]] = None,
        activation: Optional[Type[torch.nn.Module]] = None,
        normalize: bool = False,
    ) -> None:
        """Audio model.

        Args:
            conv_layers_args: Parameters of convolutions layers.
            maxpool_layers_args: Parameters of max pool layer layers.
            conv_op: Convolution operation to use. If None, defaults to torch.nn.Conv1d.
                Defaults to None.
            max_pool_op: Max pooling operation to use. If None, defaults to
                torch.nn.MaxPool1d. Defaults to None.
            activ_fn: Activation function to use. If None, defaults to
                torch.nn.LeakyReLU. Defaults to None.
            normalize: Whether to use batch normalization. Defaults to False.
        """

        super().__init__()
        self.conv_layers_args = conv_layers_args
        self.maxpool_layers_args = maxpool_layers_args
        self.conv_op = conv_op or torch.nn.Conv1d
        self.max_pool_op = max_pool_op or torch.nn.MaxPool1d
        self.activation = activation or torch.nn.LeakyReLU
        self.normalize = normalize

        network_layers = torch.nn.ModuleList()
        for conv_args, mp_args in zip(
            *[conv_layers_args.values(), maxpool_layers_args.values()], strict=True
        ):
            network_layers.extend(
                [self._conv_block(conv_args, self.activation(), self.normalize)]
            )
            network_layers.append(self.max_pool_op(**mp_args))

        self.network = torch.nn.Sequential(*network_layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters of the model."""
        for m in list(self.modules()):
            self._init_weights(m)

    def _init_weights(self, m: torch.nn.Module) -> None:
        """Helper method to initialize the parameters of the model
        with Kaiming uniform initialization.

        Args:
            m: Module to initialize.
        """

        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.kaiming_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
        if isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if "bias" in name:
                    torch.nn.init.zeros_(param)
                elif "weight" in name:
                    torch.nn.init.kaiming_uniform_(param)

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
        for conv_arg, mp_arg in zip(
            *[conv_args.values(), mp_args.values()],
            strict=True,
        ):
            # number of features in the convolution output
            layer_input = np.floor(
                (layer_input - conv_arg["kernel_size"] + 2 * conv_arg["padding"])
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
        activ_fn: torch.nn.Module,
        normalize: bool = False,
    ) -> torch.nn.Module:
        """Convolution block.

        Args:
            conv_args: Parameters of convolution layer.
            activ_fn: Activation function to use.
            normalize: Whether to use batch normalization. Defaults to False.
        """

        layer = torch.nn.ModuleList([self.conv_op(**conv_args)])

        if normalize:
            layer.append(torch.nn.BatchNorm1d(conv_args["out_channels"]))

        layer.append(activ_fn)
        return torch.nn.Sequential(*layer)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x)


class Emo18(torch.nn.Module):
    def __init__(self) -> None:
        """Speech emotion recognition model proposed in:
        https://doi.org/10.1109/ICASSP.2018.8462677
        """

        super().__init__()
        self.model, self.num_features = self.build_audio_model()

    def build_audio_model(self) -> Tuple[torch.nn.Module, int]:
        """Build the audio model: 3 blocks of convolution + max-pooling."""

        out_channels = [64, 128, 256]
        in_channels = [1]
        in_channels.extend(list(out_channels[:-1]))
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
        num_layers = len(in_channels) - 1

        num_out_features = conv_args[f"layer{num_layers}"]["out_channels"]

        return audio_model, num_out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Zhao19(torch.nn.Module):
    def __init__(self) -> None:
        """Speech emotion recognition model proposed in:
        https://doi.org/10.1016/j.bspc.2018.08.035
        """

        super().__init__()
        self.model, self.num_features = self.build_audio_model()

    def build_audio_model(self) -> Tuple[torch.nn.Module, int]:
        """Build the audio model: 3 blocks of convolution + max-pooling."""

        out_channels = [64, 64, 128, 128]
        in_channels = [1]
        in_channels.extend(list(out_channels[:-1]))
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
            conv_args,
            maxpool_args,
            normalize=True,
            activation=torch.nn.ELU,
        )

        return audio_model, out_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AudioModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> None:
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

    def _get_model(self, model_name: str) -> torch.nn.Module:
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
        transfer: Optional[Union[bool, str]] = None,
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
            transfer: Not available for this model. If set, raises an error.
                Defaults to None.
        """
        assert_no_transfer_weights(self, transfer)
        super().__init__(output_dim, None)  # no transfer learning weights
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
        self.linear = torch.nn.Linear(self.rnn.hidden_size, self.output_dim)

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, t = features.shape
        features = features.view(batch_size * seq_length, 1, t)
        audio_out = self.audio_model(features)
        audio_out = audio_out.transpose(1, 2)

        return self.rnn(audio_out)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features ((torch.Tensor) - BS x S x 1 x T)
        """
        rnn_out = self.embeddings(features)
        return self.linear(rnn_out)

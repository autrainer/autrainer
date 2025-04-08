from typing import List, Union

import torch
import torch.nn as nn

from autrainer.models.abstract_model import AbstractModel


class CRNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        dropout: Union[float, List[float]] = 0.0,
        n_cnn_layers: int = 3,
        in_channels: int = 1,
        train_cnn: bool = True,
        kernel_size: Union[int, List[int]] = 3,
        padding: Union[int, List[int]] = 1,
        stride: Union[int, List[int]] = 1,
        nb_filters: Union[int, List[int]] = 64,
        pooling: Union[List[int], List[List[int]]] = [[1, 4], [1, 4], [1, 4]],
        activation: str = "Relu",
        hidden_size: int = 64,
        n_layers_rnn: int = 1,
        rnn_type: str = "LSTM",
    ) -> None:
        """DCASE 2019 Task 4 Baseline CRNN model. For more information see:
        Paper: https://inria.hal.science/hal-02160855.
        Code: https://github.com/turpaultn/DCASE2019_task4/tree/public/baseline/models.
        Args:
            output_dim: Output dimension of the model.
            dropout: Dropout rates for the model. Can be either:
                    - A single float value (applied to all layers)
                    - A list of three float values [conv_dropout, rnn_dropout, fc_dropout]
                    Defaults to 0.0.
            n_cnn_layers: Number of CNN layers. Defaults to 3.
            in_channels: Number of input channels. Defaults to 1.
            train_cnn: Whether to train the CNN. Defaults to True.
            kernel_size: Kernel size(s) for CNN layers. Can be an int or list of ints. Defaults to 3.
            padding: Padding for CNN layers. Can be an int or list of ints. Defaults to 1.
            stride: Stride for CNN layers. Can be an int or list of ints. Defaults to 1.
            nb_filters: Number of filters for CNN layers. Can be an int or list of ints. Defaults to 64.
            pooling: Pooling for CNN layers. Can be a tuple or list of tuples. Defaults to (1, 4).
            activation: Activation function for CNN. Defaults to "Relu".
            hidden_size: Hidden size of the RNN. Defaults to 64.
            n_layers_rnn: Number of RNN layers. Defaults to 1.
            attention: Whether to use attention. Defaults to False.
            threshold: Threshold for binary classification. Defaults to 0.3.
            return_raw_predictions: If True, return the raw predictions before thresholding. Defaults to False.
            fc_grad_clip: Gradient clipping for the FC layer. Defaults to 0.5.
            rnn_type: Type of RNN to use. One of ["GRU", "LSTM"]. Defaults to "GRU".
        """
        super().__init__(output_dim)
        self.dropout = dropout
        self.n_cnn_layers = n_cnn_layers
        self.in_channels = in_channels
        self.train_cnn = train_cnn
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.nb_filters = nb_filters
        self.pooling = pooling
        self.activation = activation
        self.hidden_size = hidden_size
        self.n_layers_rnn = n_layers_rnn
        self.rnn_type = rnn_type

        if isinstance(dropout, float):
            conv_dropout = rnn_dropout = fc_dropout = dropout
        elif isinstance(dropout, list) and len(dropout) == 3:
            conv_dropout, rnn_dropout, fc_dropout = dropout
        else:
            raise ValueError(
                "dropout must be either a float or a list of three float values"
            )

        self.cnn = CNN(
            n_cnn_layers=n_cnn_layers,
            n_in_channel=in_channels,
            activation=activation,
            conv_dropout=conv_dropout,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            nb_filters=nb_filters,
            pooling=pooling,
        )

        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        cnn_output_channels = (
            nb_filters if isinstance(nb_filters, int) else nb_filters[-1]
        )

        if rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=cnn_output_channels,
                hidden_size=hidden_size,
                num_layers=n_layers_rnn,
                batch_first=True,
                bidirectional=True,
                dropout=rnn_dropout if n_layers_rnn > 1 else 0.0,
            )

        elif rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=cnn_output_channels,
                hidden_size=hidden_size,
                bidirectional=True,
                batch_first=True,
                dropout=rnn_dropout if n_layers_rnn > 1 else 0.0,
                num_layers=n_layers_rnn,
            )
        else:
            raise ValueError(
                f"RNN type {rnn_type} not supported. Use 'GRU' or 'LSTM'."
            )

        self.dropout_layer = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.cnn(features)
        bs, chan, frames, freq = x.size()
        if freq != 1:
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]
        x, _ = self.rnn(x)  # Extract output from RNN tuple
        x = self.dropout_layer(x)
        strong = self.fc(x)  # [bs, frames, nclass]
        strong = self.batch_norm(strong.transpose(1, 2)).transpose(1, 2)
        return strong

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        x = self.cnn(features)
        bs, chan, frames, freq = x.size()
        if freq != 1:
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]
        x = self.rnn(x)
        return x


class CNN(nn.Module):
    def __init__(
        self,
        n_cnn_layers: int = 3,
        n_in_channel: int = 1,
        conv_dropout: float = 0,
        kernel_size: Union[int, List[int]] = 3,
        padding: Union[int, List[int]] = 1,
        stride: Union[int, List[int]] = 1,
        nb_filters: Union[int, List[int]] = 64,
        pooling: Union[List[int], List[List[int]]] = [1, 4],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential()

        def _ensure_list(param, check_first=False):
            if check_first and isinstance(param[0], int):
                return [param] * n_cnn_layers
            elif not check_first and isinstance(param, int):
                return [param] * n_cnn_layers
            return param

        nb_filters = _ensure_list(nb_filters)

        def add_conv_layer(i: int) -> None:
            in_channels = n_in_channel if i == 0 else nb_filters[i - 1]
            out_channels = nb_filters[i]
            self.cnn.add_module(
                f"conv{i}",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    _ensure_list(kernel_size)[i],
                    _ensure_list(stride)[i],
                    _ensure_list(padding)[i],
                ),
            )
            self.cnn.add_module(
                f"batchnorm{i}",
                nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            )
            self.activation = activation.lower()
            if self.activation == "leakyrelu":
                self.cnn.add_module(f"relu{i}", nn.LeakyReLU(0.2))
            elif self.activation == "relu":
                self.cnn.add_module(f"relu{i}", nn.ReLU())
            elif self.activation == "glu":
                self.cnn.add_module(f"glu{i}", nn.GLU(out_channels))
            elif self.activation == "cg":
                self.cnn.add_module(f"cg{i}", ContextGating(out_channels))
            else:
                self.cnn.add_module(f"relu{i}", nn.ReLU())
            if conv_dropout > 0:
                self.cnn.add_module(f"dropout{i}", nn.Dropout(conv_dropout))
            self.cnn.add_module(
                f"pooling{i}",
                nn.AvgPool2d(_ensure_list(pooling, check_first=True)[i]),
            )

        for i in range(n_cnn_layers):
            add_conv_layer(i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res

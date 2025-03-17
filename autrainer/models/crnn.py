from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from autrainer.models.abstract_model import AbstractModel


# Dcase2019 Task 4 Baseline CRNN model


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
        pooling: Union[List[int], List[List[int]]] = [1, 4],
        activation: str = "Relu",
        hidden_size: int = 64,
        n_layers_rnn: int = 1,
        attention: bool = True,
        return_raw_predictions: bool = False,
        fc_grad_clip: float = 0.5,
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
        self.attention = attention
        self.return_raw_predictions = return_raw_predictions
        self.fc_grad_clip = fc_grad_clip
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
        self.rnn = BiGRU(
            n_in=cnn_output_channels,
            n_hidden=hidden_size,
            dropout=rnn_dropout if n_layers_rnn > 1 else 0.0,
            num_layers=n_layers_rnn,
        )
        self.dropout_layer = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        nn.init.xavier_normal_(self.fc.weight, gain=0.1)
        nn.init.constant_(self.fc.bias, 0.0)

        def clip_grad_hook(grad):
            return torch.clamp(grad, -self.fc_grad_clip, self.fc_grad_clip)

        self.fc.weight.register_hook(clip_grad_hook)
        self.fc.bias.register_hook(clip_grad_hook)

        if self.attention:
            self.fc_softmax = nn.Linear(hidden_size * 2, output_dim)
            self.softmax = nn.Softmax(dim=-1)
            nn.init.xavier_normal_(self.fc_softmax.weight)
            nn.init.constant_(self.fc_softmax.bias, 0.0)

    def save(self, filename: str) -> None:
        torch.save(self.state_dict(), filename)

    def forward(
        self, x: torch.Tensor, return_pre_threshold: bool = False
    ) -> Union[torch.Tensor, tuple]:
        x = self.embeddings(x)
        x = self.dropout_layer(x)
        strong = self.fc(x)
        # strong = torch.sigmoid(logits)
        if not self.attention:
            if self.return_raw_predictions or return_pre_threshold:
                return strong
            return strong.mean(1)
        sof = torch.clamp(self.softmax(self.fc_softmax(x)), min=1e-7, max=1)
        if self.return_raw_predictions or return_pre_threshold:
            return strong * sof, sof
        return (strong * sof).sum(1) / sof.sum(1)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.mean(dim=3) if freq != 1 else x.squeeze(-1)
        rnn_out = self.rnn(x)
        return rnn_out


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
                self.cnn.add_module(f"glu{i}", GLU(out_channels))
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


# FDYCRNN model


class FDYCRNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        in_channels: int = 1,
        kernel_size: Union[int, List[int]] = 3,
        stride: Union[int, List[int]] = 1,
        padding: Union[int, List[int]] = 1,
        bias: bool = False,
        n_basis_kernels: int = 4,
        temperature: float = 31.0,
        pool_dim: str = "freq",
    ) -> None:
        """FDYCRNN model. For more information see:
        Paper: https://arxiv.org/abs/2103.13444
        Code: https://github.com/frednam93/FDY-SED/blob/main/utils/model.py

        Args:
            output_dim: Output dimension of the model.
            n_cnn_layers: Number of CNN layers. Defaults to 3.
            in_channels: Number of input channels. Defaults to 1.
            kernel_size: Kernel size for CNN layers. Can be an int or list of ints. Defaults to 3.
            stride: Stride for CNN layers. Can be an int or list of ints. Defaults to 1.
            padding: Padding for CNN layers. Can be an int or list of ints. Defaults to 1.
            bias: Whether to use bias in convolutions. Defaults to False.
            n_basis_kernels: Number of basis kernels. Defaults to 4.
            temperature: Temperature for attention softmax. Defaults to 31.0.
            pool_dim: Dimension to pool attention over. One of ["freq", "time", "both", "chan"]. Defaults to "freq".
        """
        super(FDYCRNN, self).__init__(output_dim=output_dim)
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.n_basis_kernels = n_basis_kernels
        self.temperature = temperature
        self.pool_dim = pool_dim
        self.attention = Attention2d(
            in_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            n_basis_kernels,
            temperature,
            pool_dim,
        )
        self.weight = nn.Parameter(
            torch.randn(
                n_basis_kernels,
                output_dim,
                in_channels,
                self.kernel_size,
                self.kernel_size,
            ),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(n_basis_kernels, output_dim), requires_grad=True
            )
        else:
            self.bias = None
        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x):
        if self.pool_dim in ["freq", "chan"]:
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(4)
        elif self.pool_dim == "time":
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(3)
        elif self.pool_dim == "both":
            softmax_attention = (
                self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )
        batch_size = x.size(0)
        aggregate_weight = self.weight.view(
            -1, self.in_channels, self.kernel_size, self.kernel_size
        )
        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
            )
        output = output.view(
            batch_size,
            self.n_basis_kernels,
            self.output_dim,
            output.size(-2),
            output.size(-1),
        )
        if self.pool_dim in ["freq", "chan"]:
            assert softmax_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim == "time":
            assert softmax_attention.shape[-1] == output.shape[-1]
        output = torch.sum(output * softmax_attention, dim=1)
        output = output.mean(dim=(-2, -1))
        # output = torch.sigmoid(output)
        return output


class Attention2d(nn.Module):
    def __init__(
        self,
        in_planes,
        kernel_size,
        stride,
        padding,
        n_basis_kernels,
        temperature,
        pool_dim,
    ):
        super(Attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature
        hidden_planes = int(in_planes / 4)
        if hidden_planes < 4:
            hidden_planes = 4
        if not pool_dim == "both":
            self.conv1d1 = nn.Conv1d(
                in_planes,
                hidden_planes,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(
                hidden_planes, n_basis_kernels, 1, bias=True
            )
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

    def forward(self, x):
        if self.pool_dim == "freq":
            x = torch.mean(x, dim=3)
        elif self.pool_dim == "time":
            x = torch.mean(x, dim=2)
        elif self.pool_dim == "both":
            # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == "chan":
            x = torch.mean(x, dim=1)

        if not self.pool_dim == "both":
            x = self.conv1d1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

        return F.softmax(x / self.temperature, 1)


# Shared


class BiGRU(nn.Module):
    def __init__(
        self, n_in: int, n_hidden: int, dropout: float = 0, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            n_in,
            n_hidden,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(x)
        return recurrent


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


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

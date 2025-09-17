import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import requests
import torch
import torch.nn.functional as F

from autrainer.core.structs import AbstractDataBatch
from autrainer.models import AbstractModel


def _download_weights(url: str, destination: str) -> None:
    print(f"Downloading weights from {url}")
    try:
        req = requests.get(url)
        if req.status_code == 200:
            with open(destination, "wb") as f:
                f.write(req.content)
            print(f"Placed weights in {destination}")
        else:
            req.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to download weights from {url} due to {e}.") from e


def load_transfer_weights(model: torch.nn.Module, weights_link: str) -> None:
    weights_name = os.path.basename(weights_link)
    hub_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    weights_path = os.path.join(hub_dir, weights_name)
    if not os.path.exists(weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        _download_weights(weights_link, weights_path)
    if not os.path.exists(weights_path):
        raise ValueError(f"Failed to download weights to {weights_path}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location="cpu",
                weights_only=True,
            )["model"],
            strict=False,
        )


def init_layer(layer: torch.nn.Module) -> None:
    """Initialize a Linear or Convolutional layer."""
    torch.nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: torch.nn.modules.batchnorm._BatchNorm) -> None:
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm2d(self.out_channels)

        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(
        self,
        x: torch.Tensor,
        pool_size: Tuple[int, int] = (2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class ExtractLayerEmbeddings:
    def __init__(
        self,
        model: torch.nn.Module,
        layer_selector_fn: Callable = None,
    ) -> None:
        """Extract embeddings from a layer in a model.

        Args:
            model: Model to extract embeddings from.
            layer_selector_fn: Function to select the layer to extract
                embeddings from. The model is passed to this function and
                the output should be the layer. If None, the preceding layer
                of the last linear layer is selected. Defaults to None.
        """
        self.model = model
        self._embeddings = None
        self._hook_handle = None
        self._hook_layer = (
            layer_selector_fn(model)
            if layer_selector_fn is not None
            else self._find_second_to_last_layer(model)
        )

    @staticmethod
    def _find_second_to_last_layer(model: torch.nn.Module) -> torch.nn.Module:
        def _flatten_layers(
            layer: torch.nn.Module,
            layers: List[torch.nn.Module],
        ) -> None:
            if len(list(layer.children())) == 0:
                layers.append(layer)
            else:
                for child in layer.children():
                    _flatten_layers(child, layers)

        layers = []
        _flatten_layers(model, layers)
        if len(layers) < 2:
            raise ValueError("Model must have at least two layers.")
        linears = [
            idx
            for idx, layer in enumerate(layers)
            if isinstance(layer, torch.nn.Linear)
        ]
        if not linears:
            raise ValueError(
                "Cannot extract embeddings as model does not contain a linear layer."
            )
        return layers[max(linears) - 1]

    def _hook(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self._embeddings = output

    def _register(self) -> None:
        if self._hook_handle is not None:
            return
        self._hook_handle = self._hook_layer.register_forward_hook(self._hook)

    def _unregister(self) -> None:
        if self._hook_handle is None:
            return
        self._hook_handle.remove()
        self._hook_handle = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        self._register()
        self.model(x)
        self._unregister()
        return self._embeddings


def create_model_inputs(
    model: AbstractModel,
    data: AbstractDataBatch,
) -> Dict[str, torch.Tensor]:
    _forbidden = ["features", "target", "index"]
    model_inputs = {model.inputs[0]: data.features}
    for key, value in vars(data).items():
        if key not in _forbidden and key in model.inputs:
            model_inputs[key] = value
    return model_inputs


def assert_no_transfer_weights(
    model: AbstractModel,
    transfer: Optional[Union[bool, str]] = None,
) -> None:
    if not transfer:
        return
    raise ValueError(
        f"Model '{type(model).__name__}' does not support "
        f"transfer='{transfer}'. Set transfer to 'False' or 'None'. "
    )

import os
from typing import Dict, List, Optional

import torch


def format_results(
    results: Dict[str, float],
    results_type: str,
    training_type: str,
    iteration: Optional[int] = None,
) -> str:
    """Format the results of the evaluation.

    Formats the results of the evaluation into a string
    for easy printing and logging.

    Args:
        results: Dictionary containing the results of the evaluation.
        results_type: Type of results (e.g., "dev", "test").
        training_type: Type of training (e.g., "epoch", "step").
        iteration: Optional iteration number (e.g., epoch number). If None,
            the iteration is not included in the string. Defaults to None.

    Returns:
        A formatted string containing the results.
    """
    s = f"{results_type} results"
    s += f" at {training_type} {iteration}" if iteration else ""
    s += ":\n"
    max_key_len = max([len(k) for k in results])
    s += "\n".join(
        [f"{(k + ':').ljust(max_key_len + 1)} {v:.4f}" for k, v in results.items()]
    )
    return s


def load_pretrained_model_state(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    skip_last_layer: bool = True,
) -> None:
    """Load a pretrained model state dict for a model.

    Args:
        model: Model to load the state dict into.
        state_dict: State dict to load into the model.
        skip_last_layer: Whether to skip loading the state dict of the
            last linear or convolutional layer. Defaults to True.
    Raises:
        RuntimeError: If the shapes of the model and state dict do not match.
    """
    last_layer = None
    strict = True
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.modules.conv._ConvNd)):
            last_layer = name

    if last_layer is not None and skip_last_layer:
        state_dict.pop(last_layer + ".weight", None)
        state_dict.pop(last_layer + ".bias", None)
        strict = False

    model.load_state_dict(state_dict, strict=strict)


def load_pretrained_optim_state(
    optim: torch.optim.Optimizer,
    state_dict: Dict[str, torch.Tensor],
    skip_last_layer: bool = True,
) -> None:
    """Load a pretrained optimizer state dict for an optimizer.

    Args:
        optim: Optimizer to load the state dict into.
        state_dict: State dict to load into the optimizer.
        skip_last_layer: Whether to skip loading the state dict of the
            last layer. Defaults to True.
    """
    if skip_last_layer:
        keys = list(state_dict["state"].keys())
        state_dict["state"].pop(keys[-1], None)  # bias
        state_dict["state"].pop(keys[-2], None)  # weight

    optim.load_state_dict(state_dict)


def get_optimizer_params(
    model: torch.nn.Module,
    weight_decay: Optional[float] = None,
    apply_to_bias: bool = True,
    apply_to_norm: bool = True,
) -> List[Dict[str, torch.nn.Parameter]]:
    """Get the model parameters for the optimizer separated by weight decay if
    specified.

    Separates the parameters of the model into two groups:

    * Parameters that should have weight decay applied (e.g., weights of
      convolutional layers, linear layers).
    * Parameters that should not have weight decay applied (e.g., biases,
      normalization layers).

    Args:
        model: The model whose parameters are to be separated.
        weight_decay: Weight decay to apply to the parameters. If None, no
            weight decay is applied. Defaults to None.
        apply_to_bias: Whether to apply weight decay to biases.
            Defaults to True.
        apply_to_norm: Whether to apply weight decay to normalization layers.
            Defaults to True.

    Returns:
        Parameters grouped by weight decay.
    """
    try:
        rms_norm = (torch.nn.RMSNorm,)  # RMSNorm requires torch >= 2.4
    except AttributeError:
        rms_norm = ()

    norm_classes = (
        torch.nn.modules.batchnorm._NormBase,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.LocalResponseNorm,
        torch.nn.CrossMapLRN2d,
    ) + rms_norm

    if weight_decay is None:
        return [{"params": model.parameters()}]

    decay, no_decay = [], []
    for _, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if (
                isinstance(module, norm_classes)
                and not apply_to_norm
                or name == "bias"
                and not apply_to_bias
            ):
                no_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def load_checkpoint(checkpoint: str) -> Dict[str, torch.Tensor]:
    """Load a checkpoint state dict from a file.

    Args:
        checkpoint: Path to the checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.

    Returns:
        The model state dict.
    """
    # TODO: Support Hugging Face Checkpoints in the future.
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    return torch.load(checkpoint)

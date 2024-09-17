import os
from typing import Dict, Optional

import torch


def format_results(
    results: Dict[str, float],
    results_type: str,
    training_type: str,
    iteration: Optional[int] = None,
) -> str:
    s = f"{results_type} results"
    s += f" at {training_type} {iteration}" if iteration else ""
    s += ":\n"
    max_key_len = max([len(k) for k in results.keys()])
    s += "\n".join(
        [f"{(k+':').ljust(max_key_len+1)} {v:.4f}" for k, v in results.items()]
    )
    return s


def load_pretrained_model_state(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    skip_last_layer: bool = False,
) -> bool:
    """Load a pretrained model state dict for a model and skip the last linear
    layer if the output dimension of the model differs or skip_last_layer is
    True.

    Args:
        model: Model to load the state dict into.
        state_dict: State dict to load into the model.
        skip_last_layer: Whether to always skip the last linear layer when
            loading the state dict, irrespective of the shape.
            Defaults to False.

    Returns:
        True if the last linear layer was skipped, otherwise False.
    """
    last_linear_layer = None
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            last_linear_layer = name

    if last_linear_layer is None:
        raise ValueError("No linear layers found in the model.")

    module_state_dict: Dict[str, torch.Tensor] = model.state_dict()
    module_shape = module_state_dict[last_linear_layer + ".weight"].shape
    state_dict_shape = state_dict[last_linear_layer + ".weight"].shape

    if module_shape != state_dict_shape or skip_last_layer:
        state_dict.pop(last_linear_layer + ".weight", None)
        state_dict.pop(last_linear_layer + ".bias", None)
        skip_last_layer = True

    model.load_state_dict(state_dict, strict=False)
    return skip_last_layer


def load_pretrained_optim_state(
    optim: torch.optim.Optimizer,
    state_dict: Dict[str, torch.Tensor],
    skip_last_layer: bool = False,
) -> None:
    """Load a pretrained optimizer state dict for an optimizer and skip the
    last linear layer when loading the state dict.

    Args:
        optim: Optimizer to load the state dict into.
        state_dict: State dict to load into the optimizer.
        skip_last_layer: Whether to skip the last linear layer when loading
            the state dict. Defaults to False.
    """
    if skip_last_layer:
        state_len = len(state_dict["state"])
        state_dict["state"].pop(state_len - 1, None)  # bias
        state_dict["state"].pop(state_len - 2, None)  # weight

    optim.load_state_dict(state_dict)


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

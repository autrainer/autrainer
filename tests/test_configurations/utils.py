import importlib
import inspect
import os
from typing import List, Tuple, Type

from omegaconf import OmegaConf


def load_configurations(subdir: str) -> List[Tuple[str, dict]]:
    config_names = [
        f
        for f in os.listdir(f"autrainer-configurations/{subdir}")
        if f.endswith(".yaml")
    ]
    assert config_names, f"No {subdir} configurations found"
    configurations = []
    for name in config_names:
        cfg = OmegaConf.to_container(
            OmegaConf.load(f"autrainer-configurations/{subdir}/{name}")
        )
        configurations.append((name, cfg))
    return configurations


def get_class_from_import_path(import_path: str) -> Type:
    m, c = import_path.rsplit(".", 1)
    m = importlib.import_module(m)
    assert hasattr(m, c), f"Class '{c}' not found in module '{m}'"
    return getattr(m, c)


def get_required_parameters(
    test_class: Type,
    ignore_params: List[str] = None,
) -> list[str]:
    ignore_params = ignore_params or []
    return [
        k
        for k, v in inspect.signature(test_class.__init__).parameters.items()
        if v.default == inspect.Parameter.empty
        and k not in ["self", "args", "kwargs"] + ignore_params
    ]

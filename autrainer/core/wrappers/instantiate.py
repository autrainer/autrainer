from enum import Enum
import logging
from typing import Dict, Optional, Type, TypeVar, Union

import hydra
from omegaconf import DictConfig, OmegaConf


_log = logging.getLogger("autrainer.instantiate")

T = TypeVar("T")


class HydraConvertEnum(Enum):
    """Hydra conversion strategy.

    Options:
        NONE: Pass OmegaConf objects (Hydra default).
        PARTIAL: Use primitive types with Structured Configs.
        OBJECT: Use primitive types and Structured Configs as objects.
        ALL: Use primitive types only.

    For more information, see:
    https://hydra.cc/docs/advanced/instantiate_objects/overview/
    """

    NONE = "none"
    PARTIAL = "partial"
    OBJECT = "object"
    ALL = "all"


def instantiate(
    config: Union[DictConfig, Dict],
    instance_of: Optional[Type[T]] = None,
    convert: Optional[HydraConvertEnum] = None,
    recursive: bool = False,
    **kwargs,
) -> Optional[T]:
    """Instantiate an object from a configuration Dict or DictConfig.

    The config must contain a `_target_` field that specifies a relative import
    path to the object to instantiate.
    If `_target_` is None, returns None.

    Args:
        config: The configuration to instantiate.
        instance_of: The expected type of the instantiated object.
            Defaults to None.
        convert: The conversion strategy to use, one of HydraConvertEnum.
            Convert is only used if the config does not have a `_convert_`
            attribute. If None, uses HydraConvertEnum.ALL. Defaults to None.
        recursive: Whether to recursively instantiate objects. Recursive is
            only used if the config does not have a `_recursive_` field.
            Defaults to False.
        **kwargs: Additional keyword arguments to pass to the object.

    Raises:
        ValueError: If the config does not have a `_target_` field.
        ValueError: If the instantiated object is not an instance of
            `instance_of` and `instance_of` is provided.

    Returns:
        The instantiated object.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if not config.get("_target_"):
        raise ValueError(f"Missing _target_ field in config: {config}")
    if config.get("_target_") == "None":
        return None
    if not config.get("_convert_"):
        config["_convert_"] = (
            convert.value if convert else HydraConvertEnum.ALL.value
        )
    if not config.get("_recursive_"):
        config["_recursive_"] = recursive
    config.pop("id", None)

    obj = hydra.utils.instantiate(config, **kwargs)
    if instance_of is not None and not isinstance(obj, instance_of):
        instance_of_name = instance_of.__name__
        config_target = config["_target_"]
        raise ValueError(
            f"Instantiated object '{config_target}' "
            f"is not an instance of '{instance_of_name}'."
        )
    return obj


def instantiate_shorthand(
    config: Union[str, DictConfig, Dict],
    instance_of: Optional[Type[T]] = None,
    convert: Optional[HydraConvertEnum] = None,
    recursive: bool = False,
    **kwargs,
) -> Optional[T]:
    """Instantiate an object from a shorthand configuration.

    A shorthand config is either a string or a dictionary with a single key.
    If config is a string, it should be a python import path.
    If config is a dictionary, the key should be a python import path and the
    value should be a dictionary of keyword arguments.

    Args:
        config: The config to instantiate.
        instance_of: The expected type of the instantiated object.
            Defaults to None.
        convert: The conversion strategy to use, one of HydraConvertEnum.
            Convert is only used if the config does not have a `_convert_`
            attribute. If None, uses HydraConvertEnum.ALL. Defaults to None.
        recursive: Whether to recursively instantiate objects. Recursive is
            only used if the config does not have a `_recursive_` field.
            Defaults to False.
        **kwargs: Additional keyword arguments to pass to the object.

    Raises:
        ValueError: If the config is empty (None or an empty string/dictionary).

    Returns:
        The instantiated object.
    """

    def _instantiate(c: dict):
        return instantiate(c, instance_of, convert, recursive, **kwargs)

    if config is None or not config:
        raise ValueError(f"Shorthand config '{config}' is empty.")
    if isinstance(config, str):
        return _instantiate({"_target_": config})

    key = next(iter(config.keys()))
    if len(config.keys()) == 1:
        return _instantiate({"_target_": key, **(config[key] or {})})

    _log.warning(
        f"Multiple keys found in shorthand config: {config}. "
        "This is likely due to missing indentation."
    )
    config.pop(key)
    return _instantiate({"_target_": key, **config})

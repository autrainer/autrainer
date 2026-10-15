from enum import Enum
import logging
from typing import Any, TypeVar, overload

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


@overload
def instantiate(
    config: DictConfig | dict[str, Any],
    instance_of: type[T],
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> T: ...


@overload
def instantiate(
    config: DictConfig | dict[str, Any],
    instance_of: tuple[type[T], ...],
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> T: ...


@overload
def instantiate(
    config: DictConfig | dict[str, Any],
    instance_of: tuple[type[T] | None, ...],
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> T | None: ...


@overload
def instantiate(
    config: DictConfig | dict[str, Any],
    instance_of: None = None,
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> Any: ...


def instantiate(
    config: DictConfig | dict,
    instance_of: type[T] | tuple[type[T] | None, ...] | None = None,
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> Any:
    """Instantiate an object from a configuration Dict or DictConfig.

    The config must contain a `_target_` field that specifies a relative import path to
    the object to instantiate. If `_target_` is None, returns None.

    Args:
        config: The configuration to instantiate.
        instance_of: The expected type of the instantiated object. Defaults to None.
        convert: The conversion strategy to use, one of HydraConvertEnum. Convert is
            only used if the config does not have a `_convert_` attribute. If None, uses
            HydraConvertEnum.ALL. Defaults to None.
        recursive: Whether to recursively instantiate objects. Recursive is only used if
            the config does not have a `_recursive_` field. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the object.

    Raises:
        ValueError: If the config does not have a `_target_` field.
        ValueError: If the instantiated object is not an instance of `instance_of` and
            `instance_of` is provided.

    Returns:
        The instantiated object.
    """
    allow_none = (
        instance_of is None or isinstance(instance_of, tuple) and None in instance_of
    )
    if isinstance(instance_of, tuple):
        instance_of = tuple([i for i in instance_of if i is not None])
    if isinstance(config, DictConfig):
        oconfig = OmegaConf.to_container(config, resolve=True)
        if not isinstance(oconfig, dict):
            raise ValueError("Expected configuration to be a mapping.")
        config = oconfig
    if not config.get("_target_"):
        raise ValueError(f"Missing _target_ field in config: {config}")
    if config.get("_target_") == "None":
        if allow_none:
            return None
        raise ValueError("Expected non-null configuration.")
    if not config.get("_convert_"):
        config["_convert_"] = convert.value if convert else HydraConvertEnum.ALL.value
    if not config.get("_recursive_"):
        config["_recursive_"] = recursive
    config.pop("id", None)

    obj = hydra.utils.instantiate(config, **kwargs)
    if instance_of is not None and not isinstance(obj, instance_of):
        instance_of_name = (
            instance_of.__name__
            if isinstance(instance_of, type)
            else ", ".join(t.__name__ for t in instance_of)
        )
        config_target = config["_target_"]
        raise ValueError(
            f"Instantiated object '{config_target}' "
            f"is not an instance of '{instance_of_name}'."
        )
    return obj


@overload
def instantiate_shorthand(
    config: str | DictConfig | dict[str, Any],
    instance_of: type[T],
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> T: ...


@overload
def instantiate_shorthand(
    config: str | DictConfig | dict[str, Any],
    instance_of: tuple[type[T], ...],
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> T: ...


@overload
def instantiate_shorthand(
    config: str | DictConfig | dict[str, Any],
    instance_of: tuple[type[T] | None, ...],
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> T | None: ...


@overload
def instantiate_shorthand(
    config: str | DictConfig | dict[str, Any],
    instance_of: None = None,
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> Any: ...


def instantiate_shorthand(
    config: str | DictConfig | dict[str, Any],
    instance_of: type[T] | tuple[type[T] | None, ...] | None = None,
    convert: HydraConvertEnum | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> Any:
    """Instantiate an object from a shorthand configuration.

    A shorthand config is either a string or a dictionary with a single key. If config
    is a string, it should be a python import path. If config is a dictionary, the key
    should be a python import path and the value should be a dictionary of keyword
    arguments.

    Args:
        config: The config to instantiate.
        instance_of: The expected type of the instantiated object. Defaults to None.
        convert: The conversion strategy to use, one of HydraConvertEnum. Convert is
            only used if the config does not have a `_convert_` attribute. If None, uses
            HydraConvertEnum.ALL. Defaults to None.
        recursive: Whether to recursively instantiate objects. Recursive is only used if
            the config does not have a `_recursive_` field. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the object.

    Raises:
        ValueError: If the config is empty (None or an empty string/dictionary).

    Returns:
        The instantiated object.
    """

    def _instantiate(c: dict[str, Any]) -> Any:
        return instantiate(c, instance_of, convert, recursive, **kwargs)

    if config is None or not config:
        raise ValueError(f"Shorthand config '{config}' is empty.")
    if isinstance(config, str):
        return _instantiate({"_target_": config})

    key = str(next(iter(config.keys())))
    if len(config.keys()) == 1:
        return _instantiate({"_target_": key, **(config[key] or {})})

    _log.warning(
        f"Multiple keys found in shorthand config: {config}. "
        "This is likely due to missing indentation."
    )
    config.pop(key)
    ckwargs: dict[str, Any] = {str(k): v for k, v in config.items()}
    return _instantiate({"_target_": key, **ckwargs})

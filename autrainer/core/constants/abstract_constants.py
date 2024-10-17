from abc import ABC
import inspect
from typing import Any, Type


class AbstractConstants(ABC):
    """Abstract constants singleton class for managing the configurations of
    `autrainer`.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _assert_type(value: Any, value_type: Type[Any], msg: str = "") -> None:
        if not isinstance(value, value_type):
            prop = inspect.currentframe().f_back.f_code.co_qualname
            msg = f" {msg}" if msg else ""
            raise ValueError(
                f"Invalid type for '{prop}'{msg}: "
                f"expected '{value_type.__name__}', "
                f"but got '{type(value).__name__}' with value '{value}'."
            )

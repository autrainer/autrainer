from abc import ABC
from typing import Any, Type


class AbstractConstants(ABC):
    """Abstract constants singleton class for managing the configurations of
    `autrainer`.
    """

    _name = "AbstractConstants"
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _assert_type(
        self,
        value: Any,
        value_type: Type[Any],
        constant: str,
        msg: str = "",
    ) -> None:
        if not isinstance(value, value_type):
            msg = f" {msg}" if msg else ""
            raise ValueError(
                f"Invalid type for '{self._name}.{constant}'{msg}: "
                f"expected '{value_type.__name__}', "
                f"but got '{type(value).__name__}' with value '{value}'."
            )

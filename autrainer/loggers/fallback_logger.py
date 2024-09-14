import logging
from typing import Dict, Optional, Union

from omegaconf import DictConfig

from .abstract_logger import AbstractLogger


class FallbackLogger(AbstractLogger):
    def __init__(
        self,
        requested_logger: Optional[str] = None,
        extras: Optional[str] = None,
    ) -> None:
        """Fallback logger for when a requested logger is not available.

        If the requested logger is not available, a warning is logged.
        If both requested_logger and extras are None, nothing is logged.
        The logger serves as a no-op.

        Args:
            requested_logger: The requested logger. Defaults to None.
            extras: The extras required to install the logger. Defaults to None.
        """
        if requested_logger is None and extras is None:
            return
        log = logging.getLogger(__name__)
        log.warning(
            f"Requested logger '{requested_logger}' not available. "
            "Install the required extras with "
            f"'pip install autrainer[{extras}]'."
        )

    def log_params(self, params: Union[dict, DictConfig]) -> None:
        pass

    def log_and_update_metrics(
        self, metrics: Dict[str, Union[int, float]], iteration: int = None
    ) -> None:
        pass

    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        iteration=None,
    ) -> None:
        pass

    def log_timers(self, timers: Dict[str, float]) -> None:
        pass

    def log_artifact(self, filename: str, path: str = "") -> None:
        pass

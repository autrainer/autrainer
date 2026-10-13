import logging

from omegaconf import DictConfig

from .abstract_logger import AbstractLogger


class FallbackLogger(AbstractLogger):
    def __init__(
        self,
        requested_logger: str | None = None,
        extras: str | None = None,
    ) -> None:
        """Fallback logger for when a requested logger is not available.

        If the requested logger is not available, a warning is logged. If both
        requested_logger and extras are None, nothing is logged. The logger serves as a
        no-op.

        Args:
            requested_logger: The requested logger. Defaults to None.
            extras: The extras required to install the logger. Defaults to None.
        """
        if requested_logger is None and extras is None:
            return
        log = logging.getLogger(__name__)
        log.warning(
            f"Requested logger '{requested_logger}' not available. "  # noqa: G004
            "Install the required extras with "
            f"'pip install autrainer[{extras}]'."
        )

    def log_params(self, params: dict | DictConfig) -> None:
        pass

    def log_and_update_metrics(
        self,
        metrics: dict[str, int | float],
        iteration: int | None = None,
    ) -> None:
        pass

    def log_metrics(
        self,
        metrics: dict[str, int | float],
        iteration: int | None = None,
    ) -> None:
        pass

    def log_timers(self, timers: dict[str, float | str]) -> None:
        pass

    def log_artifact(self, filename: str, path: str = "") -> None:
        pass

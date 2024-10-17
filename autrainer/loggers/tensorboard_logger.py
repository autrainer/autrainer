import os
from typing import Dict, List, Union

from omegaconf import DictConfig

from autrainer.core.constants import ExportConstants
from autrainer.metrics import AbstractMetric

from .abstract_logger import (
    AbstractLogger,
    get_params_to_export,
)
from .fallback_logger import FallbackLogger


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:  # pragma: no cover
    TENSORBOARD_AVAILABLE = False  # pragma: no cover


class TensorBoardLogger(AbstractLogger):
    def __init__(
        self,
        exp_name: str,
        run_name: str,
        metrics: List[AbstractMetric],
        tracking_metric: AbstractMetric,
        artifacts: List[
            Union[str, Dict[str, str]]
        ] = ExportConstants().ARTIFACTS,
        output_dir: str = "runs",
    ) -> None:
        super().__init__(
            exp_name, run_name, metrics, tracking_metric, artifacts
        )
        self.output_dir = os.path.join(output_dir, exp_name, run_name)

    def setup(self) -> None:
        self.writer = SummaryWriter(log_dir=self.output_dir)

    def log_params(self, params: Union[dict, DictConfig]) -> None:
        for key, value in get_params_to_export(params).items():
            self.writer.add_text(key, str(value))
        self.writer.flush()

    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        iteration=None,
    ) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, iteration)
        self.writer.flush()

    def log_timers(self, timers: Dict[str, float]) -> None:
        for key, value in timers.items():
            self.writer.add_text(key, str(value))
        self.writer.flush()

    def log_artifact(self, filename: str, path: str = "") -> None:
        # TensorBoard does not support logging artifacts
        pass

    def end_run(self) -> None:
        self.writer.close()


TensorBoardLogger = (
    TensorBoardLogger
    if TENSORBOARD_AVAILABLE
    else lambda *args, **kwargs: FallbackLogger(
        "TensorBoardLogger",
        "tensorboard",
    )
)

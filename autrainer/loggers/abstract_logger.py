from abc import ABC, abstractmethod
import os
from typing import TYPE_CHECKING, Dict, List, Union

from omegaconf import DictConfig

from autrainer.core.constants import ExportConstants
from autrainer.core.utils import Timer
from autrainer.metrics import AbstractMetric


if TYPE_CHECKING:
    from autrainer.training import ModularTaskTrainer  # pragma: no cover


def get_params_to_export(
    params: Union[dict, DictConfig], prefix: str = ""
) -> Dict[str, Union[int, float, str]]:
    """Get parameters to export as a flattened dictionary filtered by
    :const:`~autrainer.core.constants.ExportConstants.IGNORE_PARAMS`
    and :const:`~autrainer.core.constants.ExportConstants.LOGGING_DEPTH`.

    Private parameters (starting with "_") are also ignored except for
    "_target_".

    Args:
        params: The parameters of the configuration.
        prefix: The prefix to add to the keys. Defaults to "".

    Returns:
        The filtered and flattened dictionary of parameters.
    """
    result = {}
    for k, v in params.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if (
            full_key in ExportConstants().IGNORE_PARAMS
            or prefix in ExportConstants().IGNORE_PARAMS
            or len(full_key.split(".")) > ExportConstants().LOGGING_DEPTH
            or prefix.startswith("_")
            or (k.startswith("_") and k != "_target_")
        ):
            continue
        if isinstance(v, (dict, DictConfig)):
            cfg_id = v.get("id", None)
            if cfg_id is None and v.get("_target_", None):
                raise KeyError(
                    f"Configuration '{full_key}' is missing an id field."
                )
            if cfg_id == "None" or not v.keys():  # placeholder syntax
                continue
            elif cfg_id is None:  # shorthand syntax
                result[full_key] = next(iter(v.keys()))
            else:  # full syntax
                result[full_key] = v.pop("id")
                result.update(get_params_to_export(v, prefix=full_key))
        else:
            result[full_key] = v
    return result


class AbstractLogger(ABC):
    def __init__(
        self,
        exp_name: str,
        run_name: str,
        metrics: List[AbstractMetric],
        tracking_metric: AbstractMetric,
        artifacts: List[
            Union[str, Dict[str, str]]
        ] = ExportConstants().ARTIFACTS,
    ) -> None:
        """Base class for loggers.

        Args:
            exp_name: The name of the experiment.
            run_name: The name of the run.
            metrics: The metrics to log.
            tracking_metric: The metric to determine the best results.
            artifacts: The artifacts to log. Defaults to
                :const:`~autrainer.core.constants.ExportConstants.ARTIFACTS`.
        """
        self.run_name = run_name
        self.exp_name = exp_name
        self.metrics = metrics
        self.tracking_metric = tracking_metric
        self.artifacts = artifacts
        self.best_metrics = {
            "train_loss.min": float("inf"),
            "dev_loss.min": float("inf"),
            "best_iteration": 0,
        }
        self.metrics_dict: Dict[str, AbstractMetric] = {}
        for metric in self.metrics:
            self.best_metrics[f"{metric.name}.{metric.suffix}"] = (
                metric.starting_metric
            )
            self.metrics_dict[metric.name] = metric

    def log_and_update_metrics(
        self, metrics: Dict[str, Union[int, float]], iteration: int = None
    ) -> None:
        """Log all metrics in the metrics dictionary for the given iteration.
        Automatically updates the best metrics based on the tracking metric.

        Args:
            metrics: The metrics to log, with the metric name as the key and
                the metric value as the value.
            iteration: The iteration, epoch, or step number.
                If None, the iteration will not be logged
                (e.g., for test metrics). Defaults to None.
        """
        if iteration:
            self._update_best_metrics(metrics, iteration)
            self.log_metrics(self.best_metrics)
        self.log_metrics(metrics, iteration)

    def _update_best_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        iteration: int,
    ) -> None:
        if self.tracking_metric.compare(
            metrics[self.tracking_metric.name],
            self.best_metrics[
                f"{self.tracking_metric.name}.{self.tracking_metric.suffix}"
            ],
        ):
            self.best_metrics["best_iteration"] = iteration
        for k, v in metrics.items():
            if ".std" in k:
                continue
            if "loss" in k:
                if v < self.best_metrics[k + ".min"]:
                    self.best_metrics[k + ".min"] = v
            else:
                metric = self.metrics_dict[k]
                if metric.compare(
                    v, self.best_metrics[f"{k}.{metric.suffix}"]
                ):
                    self.best_metrics[f"{k}.{metric.suffix}"] = v

    def setup(self) -> None:
        """Optional setup method called at the beginning of the run
        (`cb_on_train_begin`).
        """

    @abstractmethod
    def log_params(self, params: Union[dict, DictConfig]) -> None:
        """Log the parameters of the configuration.

        Args:
            params: The parameters of the configuration.
        """

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        iteration=None,
    ) -> None:
        """Log the metrics for the given iteration.

        Args:
            metrics: The metrics to log, with the metric name as the key and
                the metric value as the value.
            iteration: The iteration, epoch, or step number.
                If None, the iteration will not be logged
                (e.g., for test metrics). Defaults to None.
        """

    @abstractmethod
    def log_timers(self, timers: Dict[str, float]) -> None:
        """Log all timers (e.g., mean times for train, dev, and test).

        Args:
            timers: The timers to log, with the timer name as the key and the
                timer value as the value.
        """

    @abstractmethod
    def log_artifact(self, filename: str, path: str = "") -> None:
        """Log an artifact (e.g., a file).

        Args:
            filename: The name of the artifact.
            path: The absolute path or relative path from the current working
                directory to the artifact. Defaults to "".
        """

    def end_run(self) -> None:
        """Optional end run method called at the end of the run
        (`cb_on_train_end`).
        """

    def cb_on_train_begin(self, trainer: "ModularTaskTrainer") -> None:
        self.setup()
        self.log_params(trainer.cfg)

    def cb_on_iteration_end(
        self, trainer: "ModularTaskTrainer", iteration: int, metrics: dict
    ) -> None:
        self.log_and_update_metrics(metrics, iteration)

    def cb_on_test_end(
        self, trainer: "ModularTaskTrainer", test_results: dict
    ) -> None:
        self.log_and_update_metrics(test_results)

    def cb_on_train_end(self, trainer: "ModularTaskTrainer") -> None:
        self.log_timers(
            {
                "time." + t.timer_type + ".mean": Timer.pretty_time(
                    t.get_mean_seconds()
                )
                for t in [
                    trainer.train_timer,
                    trainer.dev_timer,
                    trainer.test_timer,
                ]
            }
        )
        for artifact in self.artifacts:
            if isinstance(artifact, dict):
                for filename, path in artifact.items():
                    self.log_artifact(
                        filename, os.path.join(trainer.output_directory, path)
                    )
            else:
                self.log_artifact(artifact, trainer.output_directory)

        self.end_run()

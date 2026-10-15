import logging
import os
from pathlib import Path

from omegaconf import DictConfig

from autrainer.metrics import AbstractMetric

from .abstract_logger import AbstractLogger, get_params_to_export
from .fallback_logger import FallbackLogger


try:
    from mlflow import MlflowClient

    logging.getLogger("mlflow.store.db.utils").setLevel(logging.WARNING)

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False  # pragma: no cover


class MLFlowLogger(AbstractLogger):
    def __init__(
        self,
        exp_name: str,
        run_name: str,
        metrics: list[AbstractMetric],
        tracking_metric: AbstractMetric,
        artifacts: list[str | dict[str, str]] | None = None,
        output_dir: str = "mlruns",
    ) -> None:
        super().__init__(exp_name, run_name, metrics, tracking_metric, artifacts)

        out = Path(output_dir).absolute()
        out.mkdir(parents=True, exist_ok=True)
        self._tracking_uri = f"sqlite:///{(out / 'mlflow.db').as_posix()}"
        self._artifact_location = (out / "artifacts").as_uri()

    def setup(self) -> None:
        self._client = MlflowClient(tracking_uri=self._tracking_uri)
        self._exp_id = self._get_or_create_experiment()
        self._delete_run_if_exists()
        self._run_id = self._client.create_run(
            self._exp_id,
            run_name=self.run_name,
        ).info.run_id

    def _get_or_create_experiment(self) -> str:
        if exp := self._client.get_experiment_by_name(self.exp_name):
            return exp.experiment_id
        return self._client.create_experiment(
            self.exp_name,
            artifact_location=self._artifact_location,
        )

    def _delete_run_if_exists(self) -> None:
        runs = self._client.search_runs(
            [self._exp_id],
            filter_string=f"tags.mlflow.runName='{self.run_name}'",
            max_results=1,
        )
        if runs:
            self._client.delete_run(runs[0].info.run_id)

    def log_params(self, params: dict | DictConfig) -> None:
        for k, v in get_params_to_export(params).items():
            self._client.log_param(self._run_id, k, v)

    def log_metrics(
        self,
        metrics: dict[str, int | float],
        iteration: int | None = None,
    ) -> None:
        for k, v in metrics.items():
            self._client.log_metric(self._run_id, k, v, step=iteration)

    def log_timers(self, timers: dict[str, float | str]) -> None:
        for k, v in timers.items():
            self._client.log_param(self._run_id, k, v)

    def log_artifact(self, filename: str, path: str = "") -> None:
        self._client.log_artifact(self._run_id, os.path.join(path, filename))

    def end_run(self) -> None:
        self._client.set_terminated(self._run_id)


MLFlowLogger = (
    MLFlowLogger
    if _MLFLOW_AVAILABLE
    else lambda *args, **kwargs: FallbackLogger(
        "MLFlowLogger", "mlflow"
    )  # pragma: no cover
)  # ty: ignore[invalid-assignment]

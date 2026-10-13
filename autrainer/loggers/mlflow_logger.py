import os
from pathlib import Path
import warnings

from omegaconf import DictConfig

from autrainer.metrics import AbstractMetric

from .abstract_logger import AbstractLogger, get_params_to_export
from .fallback_logger import FallbackLogger


try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False  # pragma: no cover


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

        self.output_dir = Path(output_dir).absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "mlflow.db"
        self.tracking_uri = f"sqlite:///{self.db_path.as_posix()}"
        self.artifact_location = (self.output_dir / "artifacts").as_uri()

    def setup(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)
        self.exp_id = self._get_or_create_experiment()
        mlflow.set_experiment(experiment_id=self.exp_id)
        self.run = self._get_or_create_run()

    def _get_or_create_experiment(self) -> str:
        experiment = mlflow.get_experiment_by_name(self.exp_name)
        if experiment:
            return experiment.experiment_id
        return mlflow.create_experiment(
            name=self.exp_name,
            artifact_location=self.artifact_location,
        )

    def _get_or_create_run(self) -> "mlflow.ActiveRun":
        self._delete_run_if_exists(self.run_name)
        return mlflow.start_run(run_name=self.run_name)

    def _delete_run_if_exists(self, run_name: str) -> None:
        client = mlflow.MlflowClient()
        runs = mlflow.search_runs(
            experiment_ids=[self.exp_id],
            filter_string=f"tags.mlflow.runName='{run_name}'",
            output_format="pandas",
        )
        if isinstance(runs, list):  # cannot happen
            return
        if runs.shape[0] > 0:
            run_id = runs.iloc[0]["run_id"]
            client.delete_run(run_id)

    def log_params(self, params: dict | DictConfig) -> None:
        params = get_params_to_export(params)
        mlflow.log_params(params)

    def log_metrics(
        self,
        metrics: dict[str, int | float],
        iteration: int | None = None,
    ) -> None:
        mlflow.log_metrics(metrics, step=iteration)

    def log_timers(self, timers: dict[str, float | str]) -> None:
        mlflow.log_params(timers)

    def log_artifact(self, filename: str, path: str = "") -> None:
        mlflow.log_artifact(os.path.join(path, filename))

    def end_run(self) -> None:
        mlflow.end_run()


MLFlowLogger = (
    MLFlowLogger
    if MLFLOW_AVAILABLE
    else lambda *args, **kwargs: FallbackLogger(
        "MLFlowLogger", "mlflow"
    )  # pragma: no cover
)  # ty: ignore[invalid-assignment]

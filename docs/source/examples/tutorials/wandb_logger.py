import os
from typing import Dict, List, Union

from omegaconf import DictConfig
import wandb

from autrainer.loggers import (
    EXPORT_ARTIFACTS,
    AbstractLogger,
    get_params_to_export,
)
from autrainer.metrics import AbstractMetric


class WandBLogger(AbstractLogger):
    def __init__(
        self,
        exp_name: str,
        run_name: str,
        metrics: List[AbstractMetric],
        tracking_metric: AbstractMetric,
        artifacts: List[Union[str, Dict[str, str]]] = EXPORT_ARTIFACTS,
        output_dir: str = "wandb",
    ) -> None:
        super().__init__(
            exp_name, run_name, metrics, tracking_metric, artifacts
        )
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def log_params(self, params: Union[dict, DictConfig]) -> None:
        wandb.init(
            project=self.exp_name,
            name=self.run_name,
            config=get_params_to_export(params),
            dir=self.output_dir,
        )

    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        iteration=None,
    ) -> None:
        wandb.log(metrics, step=iteration)

    def log_timers(self, timers: Dict[str, float]) -> None:
        wandb.log(timers)

    def log_artifact(self, filename: str, path: str = "") -> None:
        artifact = wandb.Artifact(name=filename, type="model")
        artifact.add_file(os.path.join(path, filename))
        wandb.log_artifact(artifact)

    def end_run(self) -> None:
        wandb.finish()

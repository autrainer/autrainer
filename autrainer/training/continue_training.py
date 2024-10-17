from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

import pandas as pd

from autrainer.core.constants import NamingConstants
from autrainer.postprocessing.postprocessing_utils import (
    get_run_names,
    load_yaml,
)


if TYPE_CHECKING:
    from .training import ModularTaskTrainer  # pragma: no cover


class ContinueTraining:
    def __init__(self, run_name: str, remove_continued_runs: bool = False):
        self.run_name = run_name
        self.continued_run = None
        self.remove_continued_runs = remove_continued_runs

    def cb_on_train_begin(self, trainer: "ModularTaskTrainer") -> None:
        finished_runs = get_run_names(trainer.output_directory.parent)

        current_cfg = self._create_run_config(self.run_name)
        current_iteration = int(current_cfg.pop("iterations"))

        matches = {}
        for run in finished_runs:
            metrics = os.path.join(
                trainer.output_directory.parent, run, "metrics.csv"
            )
            if not os.path.exists(metrics):
                continue

            run_cfg = self._create_run_config(run)
            run_iteration = int(run_cfg.pop("iterations"))
            if current_cfg == run_cfg and current_iteration > run_iteration:
                matches[run] = current_iteration
        if not matches:
            return
        self.continue_training(trainer, max(matches, key=matches.get))

    def cb_on_train_end(self, trainer: "ModularTaskTrainer") -> None:
        if self.continued_run is None or not self.remove_continued_runs:
            return
        shutil.rmtree(
            os.path.join(trainer.output_directory.parent, self.continued_run)
        )

    def continue_training(self, trainer: ModularTaskTrainer, run: str) -> None:
        self.continued_run = run
        self._copy_dirs(trainer, run)
        self._init_metrics(trainer, run)
        self._init_timers(trainer, run)
        self._init_states(trainer)
        self._replay_loggers(trainer)

    @staticmethod
    def _create_run_config(run: str) -> dict:
        run_values = run.split("_")
        return {
            n: v
            for n, v in zip(NamingConstants().NAMING_CONVENTION, run_values)
        }

    def _copy_dirs(self, trainer: ModularTaskTrainer, run: str) -> None:
        dirs = ["_best", "_initial"]
        for d in dirs:
            shutil.rmtree(os.path.join(trainer.output_directory, d))
        iteration_folders = [
            f
            for f in os.listdir(
                os.path.join(trainer.output_directory.parent, run)
            )
            if f.startswith(trainer.cfg.training_type)
        ]
        dirs += iteration_folders
        for dir in dirs:
            shutil.copytree(
                os.path.join(trainer.output_directory.parent, run, dir),
                os.path.join(trainer.output_directory, dir),
            )

    def _init_metrics(self, trainer: ModularTaskTrainer, run: str) -> None:
        trainer.metrics = pd.read_csv(
            os.path.join(trainer.output_directory.parent, run, "metrics.csv"),
            index_col="iteration",
        )
        trainer.initial_iteration = int(trainer.metrics.index.max() + 1)
        m = trainer.data.tracking_metric
        trainer.max_dev_metric = m.get_best(trainer.metrics[m.name])
        trainer.best_iteration = m.get_best_pos(trainer.metrics[m.name])

    def _init_timers(self, trainer: ModularTaskTrainer, run: str) -> None:
        timers = load_yaml(
            os.path.join(trainer.output_directory.parent, run, "timer.yaml")
        )
        trainer.train_timer.time_log.append(timers["train"]["total_seconds"])
        trainer.dev_timer.time_log.append(timers["dev"]["total_seconds"])
        trainer.test_timer.time_log.append(timers["test"]["total_seconds"])

    def _init_states(self, trainer: ModularTaskTrainer) -> None:
        last_iteration = int(trainer.metrics.index.max())
        last_dir = os.path.join(
            trainer.output_directory,
            f"{trainer.cfg.training_type}_{last_iteration}",
        )
        trainer.bookkeeping.load_state(trainer.model, "model.pt", last_dir)
        trainer.model = trainer.model.to(trainer.DEVICE)
        trainer.bookkeeping.load_state(
            trainer.optimizer, "optimizer.pt", last_dir
        )
        if trainer.scheduler is not None:
            trainer.bookkeeping.load_state(
                trainer.scheduler, "scheduler.pt", last_dir
            )

    def _replay_loggers(self, trainer: ModularTaskTrainer) -> None:
        for logger in trainer.loggers:
            for iteration in trainer.metrics.index:
                logger.log_and_update_metrics(
                    trainer.metrics.loc[iteration].to_dict(), iteration
                )

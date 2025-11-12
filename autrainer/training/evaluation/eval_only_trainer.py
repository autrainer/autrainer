from copy import deepcopy
import os
from pathlib import Path
from typing import List

from omegaconf import DictConfig
import pandas as pd
import torch

import autrainer
from autrainer.core.plotting import PlotMetrics
from autrainer.core.utils import (
    Bookkeeping,
    ThreadManager,
    Timer,
    save_hardware_info,
    save_requirements,
    set_device,
    set_seed,
)
from autrainer.datasets import AbstractDataset
from autrainer.loggers import AbstractLogger
from autrainer.models import AbstractModel
from autrainer.transforms import TransformManager

from ..callback_manager import CallbackManager
from ..outputs_tracker import init_trackers
from ..utils import (
    load_pretrained_model_state,
)
from .evaluator import Evaluator


class EvalOnlyTrainer:
    def __init__(self, cfg: DictConfig, output_directory: str) -> None:
        self._thread_manager = ThreadManager()
        self._cfg = cfg
        self._cfg.criterion = self._cfg.dataset.pop("criterion")

        if isinstance(self.cfg.seed, str):
            training_seed, dataset_seed = map(int, self.cfg.seed.split("-"))
        else:
            training_seed = dataset_seed = self.cfg.seed
        set_seed(training_seed)
        self.DEVICE = set_device(self.cfg.device)
        save_hardware_info(output_directory, device=self.DEVICE)
        self.output_directory = Path(output_directory)

        # ? Save current requirements.txt
        self._thread_manager.spawn(save_requirements, self.output_directory)

        # ? Load Model and Dataset Transforms and Augmentations
        model_config = self.cfg.model
        dataset_config = self.cfg.dataset

        self._loader_kwargs = {}
        for l in {"train", "dev", "test"}:
            key = f"{l}_loader_kwargs"
            self._loader_kwargs[l] = dataset_config.pop(
                key,
                self.cfg.get(key, {}),
            )

        transform_manager = TransformManager(
            model_transform=model_config.pop("transform", None),
            dataset_transform=dataset_config.pop("transform", None),
        )

        transforms = transform_manager.get_transforms()
        train_transform, dev_transform, test_transform = transforms

        # ? Load Dataset
        self.data = autrainer.instantiate(
            config=dataset_config,
            instance_of=AbstractDataset,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            seed=dataset_seed,
        )

        # ? Create Bookkeeping
        self.bookkeeping = Bookkeeping(
            output_directory=output_directory,
            file_handler_path=os.path.join(output_directory, "training.log"),
        )

        # ? Misc Training Parameters
        self.disable_progress_bar = not self.cfg.get("progress_bar", False)

        self.criterion = autrainer.instantiate_shorthand(
            config=self.cfg.criterion,
            instance_of=torch.nn.modules.loss._Loss,
            reduction="none",
        )
        if hasattr(self.criterion, "setup"):
            self.criterion.setup(self.data)
        self.criterion.to(self.DEVICE)

        # ? Load Pretrained Model, Optimizer, and Scheduler Checkpoints
        model_checkpoint = model_config.pop("model_checkpoint", None)
        if model_checkpoint and model_config.get("transfer", None):
            model_config.pop("transfer", None)  # skip transfer for checkpoints
        model_config.pop("optimizer_checkpoint", None)
        model_config.pop("scheduler_checkpoint", None)
        model_config.pop("skip_last_layer", None)

        # ? Load Model
        self.output_dim = self.data.output_dim
        self.model = autrainer.instantiate(
            config=model_config,
            instance_of=AbstractModel,
            output_dim=self.output_dim,
        )

        if model_checkpoint:
            state_dict = torch.load(
                model_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            load_pretrained_model_state(self.model, state_dict, False)

        self._thread_manager.spawn(
            self.bookkeeping.save_model_summary,
            deepcopy(self.model),
            self.data.train_dataset[0].features.unsqueeze(0).shape,
            self.DEVICE,
            "model_summary.txt",
        )

        # ? Create Dataloaders
        self.dev_loader = self.data.create_dev_loader(
            batch_size=self.cfg.inference_batch_size or self.cfg.batch_size,
            **self._loader_kwargs["dev"],
        )
        self.test_loader = self.data.create_test_loader(
            batch_size=self.cfg.inference_batch_size or self.cfg.batch_size,
            **self._loader_kwargs["test"],
        )

        # ? Take metrics from dataset and add train/dev loss
        metrics = [m.name for m in self.data.metrics] + [
            "train_loss",
            "dev_loss",
        ]
        self.metrics = pd.DataFrame(columns=metrics)
        self.max_dev_metric = self.data.tracking_metric.starting_metric
        self.best_iteration = 1

        # ? Create Timers
        self.train_timer = Timer(output_directory, "dev")  # used in loggers
        self.dev_timer = Timer(output_directory, "dev")
        self.test_timer = Timer(output_directory, "test")

        # ? Create Loggers
        self.loggers: List[AbstractLogger] = []
        for logger in self.cfg.get("loggers", []):
            self.loggers.append(
                autrainer.instantiate_shorthand(
                    config=logger,
                    instance_of=AbstractLogger,
                    exp_name=f"{self.output_directory.parent.parent.name}/eval",
                    run_name=self.output_directory.name,
                    metrics=self.data.metrics,
                    tracking_metric=self.data.tracking_metric,
                )
            )

        # ? Create Callbacks and Callback Manager
        callbacks = self.cfg.get("callbacks", [])
        self.callbacks = []
        for callback in callbacks:
            self.callbacks.append(
                autrainer.instantiate_shorthand(
                    config=callback,
                    instance_of=object,
                )
            )

        self.callback_manager = CallbackManager()
        self.callback_manager.register_multiple(
            [
                self.data,
                self.model,
                self.criterion,
                *self.loggers,
                *self.callbacks,
            ]
        )

        # ? Create Plot Metrics
        self.plot_metrics = PlotMetrics(
            self.output_directory,
            self.cfg.training_type,
            **self.cfg.plotting,
            metric_fns=self.data.metrics,
        )

        # ? Create Outputs Tracker
        self.dev_tracker, self.test_tracker = init_trackers(
            exports=[self.cfg.save_dev_outputs, self.cfg.save_test_outputs],
            prefixes=["dev", "test"],
            data=self.data,
            bookkeeping=self.bookkeeping,
        )

        # ? Create Evaluator
        self.evaluator = Evaluator()

    def eval(self) -> None:
        self.model = self.model.to(self.DEVICE)
        self.callback_manager.callback(
            position="cb_on_train_begin",
            trainer=self,
        )

        self.evaluator.dev(self, 0, track_best=False)
        os.rename(
            os.path.join(self.output_directory, f"{self.cfg.training_type}_0"),
            os.path.join(self.output_directory, "_dev"),
        )
        self.evaluator.test(self)

        self.bookkeeping.save_results_df(self.metrics, "metrics.csv")
        self.callback_manager.callback(
            position="cb_on_train_end",
            trainer=self,
        )
        self.dev_timer.save()
        self.test_timer.save()
        return self.metrics.loc[0][self.data.tracking_metric.name]

    @property
    def cfg(self) -> DictConfig:
        """Return the configuration of the trainer.

        Returns:
            Copy of the configuration.
        """
        return deepcopy(self._cfg)

from copy import deepcopy
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from tqdm import tqdm

import autrainer
from autrainer.augmentations import AugmentationManager
from autrainer.core.plotting import PlotMetrics
from autrainer.core.utils import (
    Bookkeeping,
    Timer,
    save_hardware_info,
    set_device,
    set_seed,
)
from autrainer.datasets import AbstractDataset
from autrainer.datasets.utils import AbstractFileHandler
from autrainer.loggers import AbstractLogger
from autrainer.models import AbstractModel
from autrainer.transforms import SmartCompose, TransformManager

from .callback_manager import CallbackManager
from .continue_training import ContinueTraining
from .outputs_tracker import init_trackers
from .utils import (
    disaggregated_evaluation,
    format_results,
    load_pretrained_model_state,
    load_pretrained_optim_state,
)


class ModularTaskTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        output_directory: str,
        experiment_id: str = None,
        run_name: str = None,
    ) -> None:
        """Modular Task Trainer.

        Args:
            cfg: Run configuration.
            output_directory: Output directory for the run.
            experiment_id: Experiment ID for the run. If None, the ID is
                automatically set based on the parent directory of the output
                directory. Defaults to None.
            run_name: Run name for the run. If None, the name is automatically
                set based on the output directory. Defaults to None.
        """
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
        self.initial_iteration = 1

        # ? Save current requirements.txt
        reqs_output = os.path.join(output_directory, "requirements.txt")
        with open(reqs_output, "w") as f:
            f.write(os.popen("pip freeze").read())

        # ? Load Model and Dataset Transforms and Augmentations
        model_config = self.cfg.model
        dataset_config = self.cfg.dataset

        augmentation_manager = AugmentationManager(self.cfg.augmentation)
        train_aug, dev_aug, test_aug = augmentation_manager.get_augmentations()

        transform_manager = TransformManager(
            model_transform=model_config.pop("transform", None),
            dataset_transform=dataset_config.pop("transform", None),
            train_augmentation=train_aug,
            dev_augmentation=dev_aug,
            test_augmentation=test_aug,
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
            batch_size=self.cfg.batch_size,
            inference_batch_size=self.cfg.inference_batch_size,
        )

        # ? Create Bookkeeping
        self.bookkeeping = Bookkeeping(
            output_directory=output_directory,
            file_handler_path=os.path.join(output_directory, "training.log"),
        )

        # ? Datasets and Evaluation Data
        self.train_dataset = self.data.train_dataset
        self.dev_dataset = self.data.dev_dataset
        self.test_dataset = self.data.test_dataset
        self.df_dev, self.df_test, self.stratify, self.target_transform = (
            self.data.get_evaluation_data()
        )
        self.task = self.data.task

        # ? Misc Training Parameters
        self.disable_progress_bar = not self.cfg.get("progress_bar", False)

        self.criterion = autrainer.instantiate_shorthand(
            config=self.cfg.criterion,
            instance_of=torch.nn.modules.loss._Loss,
        )
        if hasattr(self.criterion, "setup"):
            self.criterion.setup(self.data)
        self.criterion.to(self.DEVICE)

        # ? Load Pretrained Model, Optimizer, and Scheduler Checkpoints
        model_checkpoint = model_config.pop("model_checkpoint", None)
        optimizer_checkpoint = model_config.pop("optimizer_checkpoint", None)
        scheduler_checkpoint = model_config.pop("scheduler_checkpoint", None)
        skip_last_layer = model_config.pop("skip_last_layer", True)

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
            load_pretrained_model_state(
                self.model,
                state_dict,
                skip_last_layer,
            )
        self.bookkeeping.save_model_summary(
            self.model, self.train_dataset, "model_summary.txt"
        )

        # ? Load Optimizer
        self.optimizer = autrainer.instantiate(
            config=self.cfg.optimizer,
            instance_of=torch.optim.Optimizer,
            params=self.model.parameters(),
            lr=self.cfg.learning_rate,
        )
        if optimizer_checkpoint:
            state_dict = torch.load(
                optimizer_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            load_pretrained_optim_state(
                self.optimizer,
                state_dict,
                skip_last_layer,
            )

        # ? Load Scheduler
        _scheduler_cfg = self.cfg.scheduler
        self.scheduler_frequency = _scheduler_cfg.pop(
            "step_frequency", "evaluation"
        )
        if self.scheduler_frequency not in ["batch", "evaluation"]:
            raise ValueError(
                f"Scheduler frequency {self.scheduler_frequency} not supported"
            )
        self.scheduler = autrainer.instantiate(
            config=_scheduler_cfg,
            instance_of=torch.optim.lr_scheduler.LRScheduler,
            optimizer=self.optimizer,
        )
        if self.scheduler is not None and scheduler_checkpoint:
            self.scheduler.load_state_dict(
                torch.load(
                    scheduler_checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
            )

        # ? Create Dataloaders
        self.train_loader = self.data.train_loader
        self.dev_loader = self.data.dev_loader
        self.test_loader = self.data.test_loader

        # ? Take metrics from dataset and add train/dev loss
        metrics = [m.name for m in self.data.metrics] + [
            "train_loss",
            "dev_loss",
        ]
        self.metrics = pd.DataFrame(columns=metrics)
        self.max_dev_metric = self.data.tracking_metric.starting_metric
        self.best_iteration = 1

        # ? Save initial (and best) Model, Optimizer and Encoder State
        self.bookkeeping.create_folder("_initial")
        self.bookkeeping.save_state(self.model, "model.pt", "_initial")
        self.bookkeeping.save_state(self.optimizer, "optimizer.pt", "_initial")
        if self.scheduler:
            self.bookkeeping.save_state(
                self.scheduler, "scheduler.pt", "_initial"
            )

        self.bookkeeping.create_folder("_best")
        self.bookkeeping.save_state(self.model, "model.pt", "_best")
        self.bookkeeping.save_state(self.optimizer, "optimizer.pt", "_best")
        if self.scheduler:
            self.bookkeeping.save_state(
                self.scheduler, "scheduler.pt", "_best"
            )

        self.bookkeeping.save_audobject(
            self.target_transform, "target_transform.yaml"
        )
        self.bookkeeping.save_audobject(self.model, "model.yaml")
        self.bookkeeping.save_audobject(
            self.data.test_transform, "inference_transform.yaml"
        )
        self.bookkeeping.save_audobject(
            self.data.file_handler, "file_handler.yaml"
        )

        # ? Load and Save Preprocessing Pipeline if specified
        _preprocess_pipe = SmartCompose([])
        _file_handler = self.data.file_handler
        _features_subdir = cfg.dataset.get("features_subdir", "default")
        if _features_subdir != "default":
            _preprocess = OmegaConf.to_container(
                hydra.compose(f"preprocessing/{_features_subdir}")
            )["preprocessing"]
            _file_handler = autrainer.instantiate_shorthand(
                config=_preprocess["file_handler"],
                instance_of=AbstractFileHandler,
            )
            _preprocess_pipe = SmartCompose(
                [
                    autrainer.instantiate_shorthand(t)
                    for t in _preprocess["pipeline"]
                ]
            )

        self.bookkeeping.save_audobject(
            _preprocess_pipe, "preprocess_pipeline.yaml"
        )
        self.bookkeeping.save_audobject(
            _file_handler, "preprocess_file_handler.yaml"
        )

        # ? Create Timers
        self.train_timer = Timer(output_directory, "train")
        self.dev_timer = Timer(output_directory, "dev")
        self.test_timer = Timer(output_directory, "test")

        # ? Continue run
        if self.cfg.get("continue_training", False):
            self.continue_training = ContinueTraining(
                run_name=run_name or self.output_directory.name,
                remove_continued_runs=self.cfg.get(
                    "remove_continued_runs", False
                ),
            )
        else:
            self.continue_training = None

        # ? Create Loggers
        self.loggers: List[AbstractLogger] = []
        for logger in self.cfg.get("loggers", []):
            self.loggers.append(
                autrainer.instantiate_shorthand(
                    config=logger,
                    instance_of=AbstractLogger,
                    exp_name=experiment_id
                    or self.output_directory.parent.parent.name,
                    run_name=run_name or self.output_directory.name,
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
                self.optimizer,
                self.scheduler,
                self.criterion,
                self.continue_training,
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
        self.train_tracker, self.dev_tracker, self.test_tracker = (
            init_trackers(
                exports=[
                    self.cfg.save_train_outputs,
                    self.cfg.save_dev_outputs,
                    self.cfg.save_test_outputs,
                ],
                prefixes=["train", "dev", "test"],
                data=self.data,
                criterion=self.criterion,
                bookkeeping=self.bookkeeping,
            )
        )

    def train(self) -> float:
        """Train the model.

        Raises:
            ValueError: If the training type is not supported.

        Returns:
            The best value of the tracking metric.
        """
        # ? Allow optimizers to have custom step functions
        custom_step = getattr(
            self.optimizer, "custom_step", False
        ) and callable(self.optimizer.custom_step)
        self.train_step_fn = (
            self.optimizer.custom_step if custom_step else self._train_step
        )

        self.callback_manager.callback(
            position="cb_on_train_begin", trainer=self
        )

        if self.cfg.training_type == "epoch":
            self.train_epochs()
        elif self.cfg.training_type == "step":
            self.train_steps()
        else:
            raise ValueError(
                f"Training type {self.cfg.training_type} not supported"
            )

        # ? Score best model on test set
        self.bookkeeping.load_state(self.model, "model.pt", "_best")
        self.bookkeeping.load_state(self.optimizer, "optimizer.pt", "_best")
        self.model = self.model.to(self.DEVICE)
        self.model.eval()
        self.bookkeeping.create_folder("_test")
        self.test_timer.start()
        test_results = self.evaluate(
            -1,
            "_test",
            self.test_loader,
            self.df_test,
            dev_evaluation=False,
            save_to="test_holistic",
            tracker=self.test_tracker,
        )
        self.test_timer.stop()
        self.callback_manager.callback(
            position="cb_on_test_end",
            trainer=self,
            test_results=test_results,
        )
        self.metrics["iteration"] = self.metrics.index
        self.bookkeeping.save_best_results(
            self.metrics,
            "best_results.yaml",
            self.data.metrics,
            self.data.tracking_metric,
            "_best",
        )
        self.bookkeeping.log(
            format_results(
                self.metrics.loc[self.best_iteration]
                .drop("iteration")
                .to_dict(),
                "Best",
                self.cfg.training_type,
                self.best_iteration,
            )
        )
        self.bookkeeping.log(
            format_results(
                test_results,
                "Test",
                self.cfg.training_type,
            )
        )

        # ? Save Timers
        self.train_timer.save()
        self.dev_timer.save()
        self.test_timer.save()

        # ? Plot Metrics
        self.plot_metrics.plot_run(self.metrics)

        self.callback_manager.callback(
            position="cb_on_train_end", trainer=self
        )
        self.bookkeeping.save_results_df(self.metrics, "metrics.csv")
        return self.metrics.loc[self.best_iteration][
            self.data.tracking_metric.name
        ]

    def train_epochs(self):
        train_loss = []
        self.train_timer.start()
        for epoch in range(self.initial_iteration, self.cfg.iterations + 1):
            self.callback_manager.callback(
                position="cb_on_iteration_begin", trainer=self, iteration=epoch
            )
            epoch_folder = f"epoch_{epoch}"
            self.bookkeeping.create_folder(epoch_folder)
            self.model.train()
            self.model.to(self.DEVICE)
            for batch_idx, (data, target, sample_idx) in enumerate(
                tqdm(
                    self.train_loader,
                    desc="Train",
                    disable=self.disable_progress_bar,
                )
            ):
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                self.callback_manager.callback(
                    position="cb_on_step_begin",
                    trainer=self,
                    iteration=epoch,
                    batch_idx=batch_idx,
                )
                l, o = self.train_step_fn(
                    self.model,
                    data,
                    target,
                    self.criterion,
                    self.data.target_transform.probabilities_training,
                )
                if self.scheduler and self.scheduler_frequency == "batch":
                    self.scheduler.step()
                self.train_tracker.update(o, target, sample_idx)
                self.callback_manager.callback(
                    position="cb_on_step_end",
                    trainer=self,
                    iteration=epoch,
                    batch_idx=batch_idx,
                    loss=l,
                )
                train_loss.append(l)
            if epoch % self.cfg.eval_frequency == 0:
                if self.scheduler and self.scheduler_frequency == "evaluation":
                    self.scheduler.step()
                self.train_timer.stop()
                self.train_tracker.save(epoch_folder)
                train_loss = sum(train_loss) / len(train_loss)
                self.metrics.loc[epoch, "train_loss"] = train_loss
                self.dev_timer.start()
                self.evaluate(
                    epoch,
                    epoch_folder,
                    self.dev_loader,
                    self.df_dev,
                    tracker=self.dev_tracker,
                )
                self.dev_timer.stop()
                self.callback_manager.callback(
                    position="cb_on_val_end",
                    trainer=self,
                    iteration=epoch,
                    val_results=self.metrics.loc[epoch].to_dict(),
                )
                self.callback_manager.callback(
                    position="cb_on_iteration_end",
                    trainer=self,
                    iteration=epoch,
                    metrics=self.metrics.loc[epoch].to_dict(),
                )
                if epoch < self.cfg.iterations:
                    train_loss = []
                    self.train_timer.start()
            self.callback_manager.callback(
                position="cb_on_loader_exhausted",
                trainer=self,
                iteration=epoch,
            )

    def train_steps(self):
        pbar = tqdm(
            total=self.cfg.eval_frequency,
            desc="Train",
            disable=self.disable_progress_bar,
        )
        step = self.initial_iteration - 1
        self.callback_manager.callback(
            position="cb_on_iteration_begin", trainer=self, iteration=step
        )
        self.train_loader_iter = iter(self.train_loader)
        train_loss = []
        self.train_timer.start()
        while step < self.cfg.iterations:
            step += 1
            pbar.update(1)
            self.model.train()
            self.model.to(self.DEVICE)
            try:
                data, target, sample_idx = next(self.train_loader_iter)
            except StopIteration:
                self.callback_manager.callback(
                    position="cb_on_loader_exhausted",
                    trainer=self,
                    iteration=step,
                )
                self.train_loader_iter = iter(self.train_loader)
                data, target, sample_idx = next(self.train_loader_iter)
            data, target = data.to(self.DEVICE), target.to(self.DEVICE)
            self.callback_manager.callback(
                position="cb_on_step_begin",
                trainer=self,
                iteration=step,
                batch_idx=(step - 1) % self.cfg.eval_frequency,
            )
            l, o = self.train_step_fn(
                self.model,
                data,
                target,
                self.criterion,
                self.data.target_transform.probabilities_training,
            )
            if self.scheduler and self.scheduler_frequency == "batch":
                self.scheduler.step()
            self.train_tracker.update(o, target, sample_idx)
            self.callback_manager.callback(
                position="cb_on_step_end",
                trainer=self,
                iteration=step,
                batch_idx=(step - 1) % self.cfg.eval_frequency,
                loss=l,
            )
            train_loss.append(l)
            if step % self.cfg.eval_frequency == 0:
                if self.scheduler and self.scheduler_frequency == "evaluation":
                    self.scheduler.step()
                self.train_timer.stop()
                step_folder = f"step_{step}"
                self.bookkeeping.create_folder(step_folder)
                self.train_tracker.save(step_folder)
                train_loss = sum(train_loss) / len(train_loss)
                self.metrics.loc[step, "train_loss"] = train_loss
                self.dev_timer.start()
                self.evaluate(
                    step,
                    step_folder,
                    self.dev_loader,
                    self.df_dev,
                    tracker=self.dev_tracker,
                )
                self.dev_timer.stop()
                self.callback_manager.callback(
                    position="cb_on_val_end",
                    trainer=self,
                    iteration=step,
                    val_results=self.metrics.loc[step].to_dict(),
                )
                self.callback_manager.callback(
                    position="cb_on_iteration_end",
                    trainer=self,
                    iteration=step,
                    metrics=self.metrics.loc[step].to_dict(),
                )
                if step < self.cfg.iterations:
                    train_loss = []
                    pbar.reset()
                    self.train_timer.start()
                    self.callback_manager.callback(
                        position="cb_on_iteration_begin",
                        trainer=self,
                        iteration=step + 1,
                    )

    def _train_step(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        criterion: torch.nn.Module,
        probabilities_fn: Callable,
    ) -> Tuple[float, torch.Tensor]:
        self.optimizer.zero_grad()
        output = model(data)
        loss = criterion(probabilities_fn(output), target)
        loss.backward()
        self.optimizer.step()
        return loss.item(), output.detach()

    def evaluate(
        self,
        iteration: int,
        iteration_folder,
        loader,
        df,
        dev_evaluation=True,
        save_to="dev",
        tracker=None,
    ) -> Dict[str, float]:
        """Evaluate the model on the dev or test set.

        Args:
            iteration: Current iteration.
            iteration_folder: Folder to save the results to.
            loader: Dataloader to evaluate on.
            df: Groundtruth dataframe.
            dev_evaluation: Whether to evaluate on the dev set.
                Defaults to True.
            save_to: Prefix to save the results to. Defaults to "dev".
            tracker: Tracker to save the outputs. Defaults to None.

        Returns:
            Dictionary containing the evaluation results.
        """
        cb_type = "val" if dev_evaluation else "test"
        kwargs = {"iteration": iteration} if dev_evaluation else {}
        self.callback_manager.callback(
            position=f"cb_on_{cb_type}_begin",
            trainer=self,
            **kwargs,
        )
        self.model.eval()
        self.model = self.model.to(self.DEVICE)
        results = self._evaluate(
            loader=loader,
            tracker=tracker,
            iteration_folder=iteration_folder,
            cb_type=cb_type,
        )
        if dev_evaluation:
            results["dev_loss"] = results.pop("loss")
            # TODO: it's a bit ugly to filter like this
            for key in list(set(self.metrics.columns) - set(["train_loss"])):
                self.metrics.loc[iteration, key] = results[key]
        else:
            test_results = {"test_loss": results["loss"]}
            # TODO: another ugly filter
            for key in list(
                set(self.metrics.columns) - set(["train_loss", "dev_loss"])
            ):
                test_results[f"test_{key}"] = results[key]

        if dev_evaluation:
            self.bookkeeping.log(
                format_results(
                    self.metrics.loc[iteration].to_dict(),
                    "Dev",
                    self.cfg.training_type,
                    iteration,
                )
            )

        logging_results = disaggregated_evaluation(
            targets=tracker.targets,
            predictions=tracker.predictions,
            indices=tracker.indices,
            groundtruth=df,
            metrics=self.data.metrics,
            target_column=self.data.target_column,
            stratify=self.stratify,
        )
        if dev_evaluation:
            logging_results["dev_loss"] = {"all": results["dev_loss"]}
            logging_results["iteration"] = iteration
        else:
            logging_results["loss"] = {"all": results["loss"]}

        self.bookkeeping.save_results_dict(
            logging_results, save_to + ".yaml", iteration_folder
        )

        if not dev_evaluation:
            return test_results

        if self.data.tracking_metric.compare(
            results[self.data.tracking_metric.name], self.max_dev_metric
        ):
            self.max_dev_metric = results[self.data.tracking_metric.name]
            self.best_iteration = iteration
            self.bookkeeping.save_state(self.model, "model.pt", "_best")
            self.bookkeeping.save_state(
                self.optimizer, "optimizer.pt", "_best"
            )
            if self.scheduler:
                self.bookkeeping.save_state(
                    self.scheduler, "scheduler.pt", "_best"
                )

            # ? additionally save all best results
            tracker.save("_best", reset=False)
            self.bookkeeping.save_results_dict(
                logging_results, "dev.yaml", "_best"
            )

        if (
            iteration % self.cfg.save_frequency == 0
            or iteration == self.cfg.iterations
        ):
            self.bookkeeping.save_state(
                self.model, "model.pt", iteration_folder
            )
            self.bookkeeping.save_state(
                self.optimizer, "optimizer.pt", iteration_folder
            )
            if self.scheduler:
                self.bookkeeping.save_state(
                    self.scheduler, "scheduler.pt", iteration_folder
                )
        tracker.reset()

    def _evaluate(
        self, loader, tracker, iteration_folder, cb_type: str = "val"
    ):
        with torch.no_grad():
            loss = 0
            for batch_idx, (features, target, sample_idx) in enumerate(
                tqdm(
                    loader,
                    desc="Evaluate" if cb_type == "val" else "Test",
                    disable=self.disable_progress_bar,
                )
            ):
                self.callback_manager.callback(
                    position=f"cb_on_{cb_type}_step_begin",
                    trainer=self,
                    batch_idx=batch_idx,
                )
                output = self.model(features.to(self.DEVICE))
                probabilities_fn = (
                    self.data.target_transform.probabilities_training
                )
                loss += (
                    self.criterion(
                        probabilities_fn(output),
                        target.to(self.DEVICE),
                    )
                    .cpu()
                    .item()
                )
                tracker.update(output, target, sample_idx)
                self.callback_manager.callback(
                    position=f"cb_on_{cb_type}_step_end",
                    trainer=self,
                    batch_idx=batch_idx,
                    loss=loss,
                )
            loss /= len(loader) + 1
        tracker.save(iteration_folder, reset=False)
        results = {
            "loss": loss,
        }
        for metric in self.data.metrics:
            results[metric.name] = metric(tracker.targets, tracker.predictions)
        return results

    @property
    def cfg(self) -> DictConfig:
        """Return the configuration of the trainer.

        Returns:
            Copy of the configuration.
        """
        return deepcopy(self._cfg)

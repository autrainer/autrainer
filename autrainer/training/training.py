from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from tqdm import tqdm

import autrainer
from autrainer.augmentations import AugmentationManager
from autrainer.core.plotting import PlotMetrics
from autrainer.core.structs import AbstractDataBatch
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
from autrainer.datasets.utils import AbstractFileHandler, AudioFileHandler
from autrainer.loggers import AbstractLogger
from autrainer.models import AbstractModel
from autrainer.models.utils import create_model_inputs
from autrainer.transforms import SmartCompose, TransformManager

from .callback_manager import CallbackManager
from .continue_training import ContinueTraining
from .outputs_tracker import OutputsTracker, init_trackers
from .utils import (
    disaggregated_evaluation,
    format_results,
    get_optimizer_params,
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
        """Trainer managing the training of a model given a configuration.

        Args:
            cfg: Run configuration.
            output_directory: Output directory for the run.
            experiment_id: Experiment ID for the run. If None, the ID is
                automatically set based on the parent directory of the output
                directory. Defaults to None.
            run_name: Run name for the run. If None, the name is automatically
                set based on the output directory. Defaults to None.
        """
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
        self.initial_iteration = 1

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

        self._thread_manager.spawn(
            self.bookkeeping.save_model_summary,
            deepcopy(self.model),
            self.data.train_dataset[0].features.unsqueeze(0).shape,
            self.DEVICE,
            "model_summary.txt",
        )

        # ? Load Optimizer
        optimizer_cfg = self.cfg.optimizer
        _wd = optimizer_cfg.pop("weight_decay", None)
        _wd_bias = optimizer_cfg.pop("apply_weight_decay_to_bias", False)
        _wd_norm = optimizer_cfg.pop("apply_weight_decay_to_norm", False)

        self.optimizer = autrainer.instantiate(
            config=optimizer_cfg,
            instance_of=torch.optim.Optimizer,
            params=get_optimizer_params(self.model, _wd, _wd_bias, _wd_norm),
            lr=self.cfg.learning_rate,
        )
        if optimizer_checkpoint:
            state_dict = torch.load(
                optimizer_checkpoint,
                map_location=self.DEVICE,
                weights_only=True,
            )
            load_pretrained_optim_state(
                self.optimizer,
                state_dict,
                skip_last_layer,
            )

        # ? Load Scheduler
        if self.cfg.scheduler.id != "None":
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
        else:
            self.scheduler = None

        # ? Create Dataloaders
        self.train_loader = self.data.create_train_loader(
            batch_size=self.cfg.batch_size,
            **self._loader_kwargs["train"],
        )
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

        # ? Save initial (and best) Model, Optimizer and Scheduler states
        save_tasks = [
            (self.model, "model.pt", "_initial"),
            (self.optimizer, "optimizer.pt", "_initial"),
            (self.scheduler, "scheduler.pt", "_initial"),
            (self.model, "model.pt", "_best"),
            (self.optimizer, "optimizer.pt", "_best"),
            (self.scheduler, "scheduler.pt", "_best"),
        ]
        for task in save_tasks:
            self._thread_manager.spawn(self.bookkeeping.save_state, *task)

        # ? Load and Save Preprocessing Pipeline if specified
        _preprocess_pipe = SmartCompose([])
        _file_handler = self.data.file_handler
        _features_subdir = cfg.dataset.get("features_subdir", "default")
        if (
            not isinstance(self.data.file_handler, AudioFileHandler)
            and _features_subdir != "default"
        ):
            _preprocess = OmegaConf.to_container(
                hydra.compose(f"preprocessing/{_features_subdir}")
            )["preprocessing"]
            _file_handler = autrainer.instantiate_shorthand(
                config=_preprocess["file_handler"],
                instance_of=AbstractFileHandler,
            )
            _preprocess_pipe = SmartCompose(
                [autrainer.instantiate_shorthand(t) for t in _preprocess["pipeline"]]
            )

        save_tasks = [
            (self.data.target_transform, "target_transform.yaml"),
            (self.model, "model.yaml"),
            (self.data.test_transform, "inference_transform.yaml"),
            (self.data.file_handler, "file_handler.yaml"),
            (_preprocess_pipe, "preprocess_pipeline.yaml"),
            (_file_handler, "preprocess_file_handler.yaml"),
        ]
        for task in save_tasks:
            self._thread_manager.spawn(self.bookkeeping.save_audobject, *task)

        # ? Create Timers
        self.train_timer = Timer(output_directory, "train")
        self.dev_timer = Timer(output_directory, "dev")
        self.test_timer = Timer(output_directory, "test")

        # ? Continue run
        if self.cfg.get("continue_training", False):
            self.continue_training = ContinueTraining(
                run_name=run_name or self.output_directory.name,
                remove_continued_runs=self.cfg.get("remove_continued_runs", False),
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
                    exp_name=experiment_id or self.output_directory.parent.parent.name,
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
                *self.loggers,
                *self.callbacks,
                self.continue_training,  # has to be last as it might overwrite other callbacks  # noqa: E501
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
        if self.cfg.save_train_outputs:
            self.train_tracker = OutputsTracker(
                export=True,
                prefix="train",
                data=self.data,
                bookkeeping=self.bookkeeping,
            )
        else:
            self.train_tracker = None

    def train(self) -> float:
        """Train the model.

        Raises:
            ValueError: If the training type is not supported.

        Returns:
            The best value of the tracking metric.
        """
        # ? Allow optimizers to have custom step functions
        custom_step = getattr(self.optimizer, "custom_step", False) and callable(
            self.optimizer.custom_step
        )
        self.train_step_fn = (
            self.optimizer.custom_step if custom_step else self._train_step
        )

        self._thread_manager.join()
        self.callback_manager.callback(position="cb_on_train_begin", trainer=self)

        if self.cfg.training_type == "epoch":
            self.train_epochs()
        elif self.cfg.training_type == "step":
            self.train_steps()
        else:
            raise ValueError(f"Training type {self.cfg.training_type} not supported")

        # ? Score best model on test set
        self.bookkeeping.load_state(self.model, "model.pt", "_best")
        self.bookkeeping.load_state(self.optimizer, "optimizer.pt", "_best")
        self.model.to(self.DEVICE)
        self.model.eval()
        self.bookkeeping.create_folder("_test")
        self.test_timer.start()
        test_results = self.evaluate(
            -1,
            "_test",
            self.test_loader,
            self.data.df_test,
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
                self.metrics.loc[self.best_iteration].drop("iteration").to_dict(),
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

        self.bookkeeping.save_results_df(self.metrics, "metrics.csv")
        self.callback_manager.callback(position="cb_on_train_end", trainer=self)
        return self.metrics.loc[self.best_iteration][self.data.tracking_metric.name]

    def train_epochs(self) -> None:
        """Train the model for a fixed number of epochs."""
        train_loss = []
        pm = (
            self._loader_kwargs["train"].get("pin_memory", False)
            and self.DEVICE.type == "cuda"
        )
        self.train_timer.start()
        for epoch in range(self.initial_iteration, self.cfg.iterations + 1):
            self.callback_manager.callback(
                position="cb_on_iteration_begin", trainer=self, iteration=epoch
            )
            epoch_folder = f"epoch_{epoch}"
            self.bookkeeping.create_folder(epoch_folder)
            self.model.train()
            self.model.to(self.DEVICE)
            for batch_idx, data in enumerate(
                tqdm(
                    self.train_loader,
                    desc="Train",
                    disable=self.disable_progress_bar,
                )
            ):
                data.to(self.DEVICE, non_blocking=pm)
                self.callback_manager.callback(
                    position="cb_on_step_begin",
                    trainer=self,
                    iteration=epoch,
                    batch_idx=batch_idx,
                )
                loss, output = self.train_step_fn(
                    self.model,
                    data,
                    self.criterion,
                    self.data.target_transform.probabilities_training,
                )
                loss = loss.detach()
                output = output.detach()
                reduced_loss = loss.mean().item()
                if self.scheduler and self.scheduler_frequency == "batch":
                    self.scheduler.step()
                if self.train_tracker:
                    self.train_tracker.update(output, data.target, loss, data.index)
                self.callback_manager.callback(
                    position="cb_on_step_end",
                    trainer=self,
                    iteration=epoch,
                    batch_idx=batch_idx,
                    loss=reduced_loss,
                )
                train_loss.append(reduced_loss)
            if epoch % self.cfg.eval_frequency == 0:
                if self.scheduler and self.scheduler_frequency == "evaluation":
                    self.scheduler.step()
                self.train_timer.stop()
                if self.train_tracker:
                    self.train_tracker.save(epoch_folder)
                train_loss = sum(train_loss) / len(train_loss)
                self.metrics.loc[epoch, "train_loss"] = train_loss
                self.dev_timer.start()
                self.evaluate(
                    epoch,
                    epoch_folder,
                    self.dev_loader,
                    self.data.df_dev,
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

    def train_steps(self) -> None:
        """Train the model for a fixed number of steps."""
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
        pm = (
            self._loader_kwargs["train"].get("pin_memory", False)
            and self.DEVICE.type == "cuda"
        )
        self.train_timer.start()
        while step < self.cfg.iterations:
            step += 1
            pbar.update(1)
            self.model.train()
            self.model.to(self.DEVICE)
            try:
                data = next(self.train_loader_iter)
            except StopIteration:
                self.callback_manager.callback(
                    position="cb_on_loader_exhausted",
                    trainer=self,
                    iteration=step,
                )
                self.train_loader_iter = iter(self.train_loader)
                data = next(self.train_loader_iter)
            data.to(self.DEVICE, non_blocking=pm)
            self.callback_manager.callback(
                position="cb_on_step_begin",
                trainer=self,
                iteration=step,
                batch_idx=(step - 1) % self.cfg.eval_frequency,
            )
            loss, output = self.train_step_fn(
                self.model,
                data,
                self.criterion,
                self.data.target_transform.probabilities_training,
            )
            loss = loss.detach()
            output = output.detach()
            reduced_loss = loss.mean().item()
            if self.scheduler and self.scheduler_frequency == "batch":
                self.scheduler.step()
            if self.train_tracker:
                self.train_tracker.update(output, data.target, loss, data.index)
            self.callback_manager.callback(
                position="cb_on_step_end",
                trainer=self,
                iteration=step,
                batch_idx=(step - 1) % self.cfg.eval_frequency,
                loss=reduced_loss,
            )
            train_loss.append(reduced_loss)
            if step % self.cfg.eval_frequency == 0:
                if self.scheduler and self.scheduler_frequency == "evaluation":
                    self.scheduler.step()
                self.train_timer.stop()
                step_folder = f"step_{step}"
                self.bookkeeping.create_folder(step_folder)
                if self.train_tracker:
                    self.train_tracker.save(step_folder)
                train_loss = sum(train_loss) / len(train_loss)
                self.metrics.loc[step, "train_loss"] = train_loss
                self.dev_timer.start()
                self.evaluate(
                    step,
                    step_folder,
                    self.dev_loader,
                    self.data.df_dev,
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
        model: AbstractModel,
        data: AbstractDataBatch,
        criterion: torch.nn.Module,
        probabilities_fn: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model for a single step.

        Args:
            model: Model to train.
            data: Data to train on.
            criterion: Criterion to optimize.
            probabilities_fn: Function to convert model output to
                probabilities.

        Returns:
            Tuple containing the non-reduced loss and model outputs.
        """
        self.optimizer.zero_grad()
        output = model(**create_model_inputs(model, data))
        loss = criterion(probabilities_fn(output), data.target)
        loss.mean().backward()
        self.optimizer.step()
        return loss, output

    def evaluate(
        self,
        iteration: int,
        iteration_folder: str,
        loader: torch.utils.data.DataLoader,
        df: pd.DataFrame,
        dev_evaluation: bool = True,
        save_to: str = "dev",
        tracker: Optional[OutputsTracker] = None,
    ) -> Optional[Dict[str, float]]:
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
            Dictionary containing the evaluation results if evaluating on the test set,
            otherwise None.
        """
        cb_type = "val" if dev_evaluation else "test"
        kwargs = {"iteration": iteration} if dev_evaluation else {}
        self.callback_manager.callback(
            position=f"cb_on_{cb_type}_begin",
            trainer=self,
            **kwargs,
        )
        self.model.eval()
        self.model.to(self.DEVICE)
        lk = self._loader_kwargs["dev" if dev_evaluation else "test"]
        results = self._evaluate(
            loader=loader,
            tracker=tracker,
            iteration_folder=iteration_folder,
            loader_kwargs=lk,
            cb_type=cb_type,
        )
        if dev_evaluation:
            results["dev_loss"] = results.pop("loss")
            # TODO: it's a bit ugly to filter like this
            for key in list(set(self.metrics.columns) - {"train_loss"}):
                self.metrics.loc[iteration, key] = results[key]
        else:
            test_results = {"test_loss": results["loss"]}
            # TODO: another ugly filter
            for key in list(set(self.metrics.columns) - {"train_loss", "dev_loss"}):
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
        if self.data.stratify or isinstance(self.data.target_column, list):
            logging_results = disaggregated_evaluation(
                targets=tracker.targets,
                predictions=tracker.predictions,
                indices=tracker.indices,
                groundtruth=df,
                metrics=self.data.metrics,
                target_column=self.data.target_column,
                stratify=self.data.stratify,
            )
        else:
            logging_results = {
                k: {"all": v} for k, v in results.items() if not k.endswith("loss")
            }
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
            self.bookkeeping.save_state(self.optimizer, "optimizer.pt", "_best")
            if self.scheduler:
                self.bookkeeping.save_state(self.scheduler, "scheduler.pt", "_best")

            # ? additionally save all best results
            tracker.save("_best", reset=False)
            self.bookkeeping.save_results_dict(logging_results, "dev.yaml", "_best")

        if iteration % self.cfg.save_frequency == 0 or iteration == self.cfg.iterations:
            self.bookkeeping.save_state(self.model, "model.pt", iteration_folder)
            self.bookkeeping.save_state(
                self.optimizer, "optimizer.pt", iteration_folder
            )
            if self.scheduler:
                self.bookkeeping.save_state(
                    self.scheduler, "scheduler.pt", iteration_folder
                )
        tracker.reset()
        return None

    def _evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        tracker: OutputsTracker,
        iteration_folder: str,
        loader_kwargs: Dict[str, Any],
        cb_type: str = "val",
    ) -> Dict[str, float]:
        """Evaluate the model on the dev or test set.

        Args:
            loader: Dataloader to evaluate on.
            tracker: Tracker to save the outputs.
            iteration_folder: Iteration folder to save the results to.
            cb_type: Callback type. Defaults to "val".

        Returns:
            Dictionary containing the results on the dev or test set.
        """
        pm = loader_kwargs.get("pin_memory", False) and self.DEVICE.type == "cuda"
        with torch.no_grad():
            losses = 0
            for batch_idx, data in enumerate(
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
                data.to(self.DEVICE, non_blocking=pm)
                output = self.model(**create_model_inputs(self.model, data))
                loss = self.criterion(
                    self.data.target_transform.probabilities_training(output),
                    data.target,
                )
                reduced_loss = loss.mean().item()
                losses += reduced_loss
                tracker.update(output, data.target, loss, data.index)
                self.callback_manager.callback(
                    position=f"cb_on_{cb_type}_step_end",
                    trainer=self,
                    batch_idx=batch_idx,
                    loss=reduced_loss,
                )
            losses /= len(loader)
        tracker.save(iteration_folder, reset=False)
        results = {
            "loss": losses,
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

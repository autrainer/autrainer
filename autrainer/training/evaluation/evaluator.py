from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from autrainer.metrics import AbstractMetric
from autrainer.models.utils import create_model_inputs

from ..outputs_tracker import OutputsTracker
from ..utils import format_results


if TYPE_CHECKING:
    from autrainer.core.structs import AbstractDataBatch

    from ..trainer import Trainer


class Evaluator:
    def dev(
        self,
        trainer: "Trainer",
        iteration: int,
        track_best: bool = True,
    ) -> None:
        """Evaluate the model on the dev set.

        Args:
            trainer: Trainer instance.
            iteration: Current iteration.
            track_best: Whether dev is called during training and the best iteration
                should be tracked. Defaults to True.
        """
        trainer.dev_timer.start()
        iteration_dir = f"{trainer.cfg.training_type}_{iteration}"
        results, logging_results = self._evaluate_dev_test(
            trainer=trainer,
            iteration=iteration,
            iteration_dir=iteration_dir,
            df=trainer.data.df_dev,
            tracker=trainer.dev_tracker,
        )

        for k, v in results.items():
            trainer.metrics.loc[iteration, k] = v
        trainer.bookkeeping.log(
            format_results(
                trainer.metrics.loc[iteration].to_dict(),
                "Dev",
                trainer.cfg.training_type,
                iteration,
            )
        )

        logging_results["dev_loss"] = results["dev_loss"]
        logging_results["iteration"] = iteration

        if track_best and (
            trainer.data.tracking_metric.compare(
                results[trainer.data.tracking_metric.name], trainer.max_dev_metric
            )
            or getattr(trainer, "_first_logging_iteration", True)
        ):
            trainer._first_logging_iteration = False
            trainer.max_dev_metric = results[trainer.data.tracking_metric.name]
            trainer.best_iteration = iteration
            trainer.bookkeeping.save_state(trainer.model, "model.pt", "_best")
            trainer.bookkeeping.save_state(trainer.optimizer, "optimizer.pt", "_best")
            if trainer.scheduler:
                trainer.bookkeeping.save_state(
                    trainer.scheduler, "scheduler.pt", "_best"
                )

            # ? additionally save all best results
            trainer.dev_tracker.save("_best", reset=False)
            trainer.bookkeeping.save_results_dict(logging_results, "dev.yaml", "_best")

        if track_best and (
            iteration % trainer.cfg.save_frequency == 0
            or iteration == trainer.cfg.iterations
        ):
            trainer.bookkeeping.save_state(trainer.model, "model.pt", iteration_dir)
            trainer.bookkeeping.save_state(
                trainer.optimizer, "optimizer.pt", iteration_dir
            )
            if trainer.scheduler:
                trainer.bookkeeping.save_state(
                    trainer.scheduler, "scheduler.pt", iteration_dir
                )
        trainer.dev_tracker.reset()
        trainer.dev_timer.stop()

        trainer.callback_manager.callback(
            position="cb_on_val_end",
            trainer=trainer,
            iteration=iteration,
            val_results=trainer.metrics.loc[iteration].to_dict(),
        )
        trainer.bookkeeping.save_results_dict(
            logging_results, "dev.yaml", iteration_dir
        )

    def test(self, trainer: "Trainer") -> None:
        """Evaluate the model on the test set.

        Args:
            trainer: Trainer instance.
        """
        trainer.test_timer.start()
        results, logging_results = self._evaluate_dev_test(
            trainer=trainer,
            iteration=-1,
            iteration_dir="_test",
            df=trainer.data.df_test,
            tracker=trainer.test_tracker,
        )
        results = {f"test_{k}": v for k, v in results.items()}
        logging_results["loss"] = {"all": results["test_loss"]}

        trainer.test_tracker.reset()
        trainer.test_timer.stop()
        trainer.callback_manager.callback(
            position="cb_on_test_end",
            trainer=trainer,
            test_results=results,
        )
        trainer.bookkeeping.save_results_dict(
            logging_results,
            "test_holistic.yaml",
            "_test",
        )
        trainer.bookkeeping.log(
            format_results(results, "Test", trainer.cfg.training_type)
        )

    def _evaluate_dev_test(
        self,
        trainer: "Trainer",
        iteration: int,
        iteration_dir: str,
        df: pd.DataFrame,
        tracker: OutputsTracker,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Evaluate the model on the dev or test set.

        Args:
            iteration: Current iteration.
            iteration_dir: Directory to save the results to.
            df: Groundtruth dataframe.
            tracker: Tracker to save the outputs.

        Returns:
            Tuple containing the evaluation results and (stratified) logging results.
        """
        dev_evaluation = iteration_dir != "_test"
        cb_type = "val" if dev_evaluation else "test"
        kwargs = {"iteration": iteration} if dev_evaluation else {}
        trainer.model.eval()
        trainer.model = trainer.model.to(trainer.DEVICE)
        trainer.callback_manager.callback(
            position=f"cb_on_{cb_type}_begin",
            trainer=trainer,
            **kwargs,
        )
        _prefix = "dev" if dev_evaluation else "test"
        results = self._evaluate(
            trainer=trainer,
            iteration=iteration,
            tracker=tracker,
            iteration_dir=iteration_dir,
            loader_kwargs=trainer._loader_kwargs[_prefix],
            cb_type=cb_type,
        )

        if trainer.data.stratify or isinstance(trainer.data.target_column, list):
            logging_results = self._disaggregated(
                targets=tracker.targets,
                predictions=tracker.predictions,
                indices=tracker.indices,
                groundtruth=df,
                metrics=trainer.data.metrics,
                target_column=trainer.data.target_column,
                stratify=trainer.data.stratify,
            )
        else:
            logging_results = {}
        for k, v in results.items():
            if k.endswith("loss"):
                continue
            if k not in logging_results:
                logging_results[k] = {}
            logging_results[k]["all"] = v

        return results, logging_results

    @torch.no_grad()
    def _evaluate(
        self,
        trainer: "Trainer",
        iteration: int,
        tracker: OutputsTracker,
        iteration_dir: str,
        loader_kwargs: Dict[str, Any],
        cb_type: str = "val",
    ) -> Dict[str, float]:
        """Evaluate the model on the dev or test set.

        Args:
            trainer: Trainer intstance.
            iteration: Current iteration.
            tracker: Tracker to save the outputs.
            cb_type: Callback type. Defaults to "val".

        Returns:
            Dictionary containing the results on the dev or test set.
        """
        desc = "Evaluate" if cb_type == "val" else "Test"
        dis = trainer.disable_progress_bar
        loader = trainer.dev_loader if cb_type == "val" else trainer.test_loader
        pm = loader_kwargs.get("pin_memory", False) and trainer.DEVICE.type == "cuda"
        prob_fn = trainer.data.target_transform.probabilities_training
        losses = 0
        data: AbstractDataBatch
        for batch_idx, data in enumerate(tqdm(loader, desc=desc, disable=dis)):
            data.to(trainer.DEVICE, non_blocking=pm)
            trainer.callback_manager.callback(
                position=f"cb_on_{cb_type}_step_begin",
                trainer=trainer,
                batch_idx=batch_idx,
            )
            output = trainer.model(**create_model_inputs(trainer.model, data))
            loss = trainer.criterion(prob_fn(output), data.target)
            reduced_loss = loss.mean().item()
            losses += reduced_loss
            tracker.update(output, data.target, loss, data.index)
            cb_kwargs = {"iteration": iteration} if cb_type == "val" else {}
            trainer.callback_manager.callback(
                position=f"cb_on_{cb_type}_step_end",
                trainer=trainer,
                batch_idx=batch_idx,
                loss=reduced_loss,
                **cb_kwargs,
            )
        tracker.save(iteration_dir, reset=False)
        results = {
            metric.name: metric(tracker.targets, tracker.predictions)
            for metric in trainer.data.metrics
        }
        results["dev_loss" if cb_type == "val" else "loss"] = losses / len(loader)
        return results

    @staticmethod
    def _disaggregated(
        targets: np.ndarray,
        predictions: np.ndarray,
        indices: np.ndarray,
        groundtruth: pd.DataFrame,
        metrics: List[AbstractMetric],
        target_column: Union[str, List[str]],
        stratify: List[str] = None,
    ) -> Dict:
        r"""Runs evaluation, optionally disaggregated.

        Computes each metric globally (over all targets)
        and unitary (over each target).
        Additionally supports disaggregated evaluations
        for different values
        of columns present in the data dataframe.

        Args:
            targets: array with groundtruth values.
            predictions: array with model predictions.
            indices: array tracking initial dataset indices.
            groundruth: dataframe with groundtruth data and metadata.
            metrics: list of metrics to use for evaluation.
            target_column: columns to evaluate on.
            stratify: optional list of metadata to run evaluation
                in stratified manner.

        Returns:
            Dictionary containing the results of the disaggregated evaluation.

        """
        results = {m.name: {} for m in metrics}
        for metric in metrics:
            if isinstance(target_column, list):
                # this handles the case of multi-label classification and
                # multi-target regression
                # loops over all targets and computes the metric for them
                for idx, col in enumerate(target_column):
                    results[metric.name][col] = metric.unitary(
                        targets[:, idx],
                        predictions[:, idx],
                    )
            results[metric.name]["all"] = metric(targets, predictions)
            for s in stratify:
                if isinstance(target_column, list):
                    raise ValueError(
                        "Stratified evaluation not supported for multi-label "
                        "classification and multi-target regression."
                    )
                for v in groundtruth[s].unique():
                    idx = groundtruth.loc[groundtruth[s] == v].index
                    # Map groundtruth indices to tracker indices
                    # This accounts for random shuffling
                    mapped_indices = [i for i, x in enumerate(indices) if x in idx]
                    results[metric.name][v] = metric(
                        targets[mapped_indices],
                        predictions[mapped_indices],
                    )
        return results

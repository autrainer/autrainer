import logging
import os
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import warnings

import audobject
import numpy as np
import pandas as pd
import torch
from torchinfo import summary
import yaml

from autrainer.metrics import AbstractMetric


if TYPE_CHECKING:
    from autrainer.datasets import AbstractDataset  # pragma: no cover


class Bookkeeping:
    def __init__(
        self,
        output_directory: str,
        file_handler_path: Optional[str] = None,
    ) -> None:
        """Bookkeeping to handle general disk operations and interactions.

        Args:
            output_directory: Output directory to save files to.
            file_handler_path: Path to save the log file to. Defaults to None.
        """
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
        self.original_stdout = sys.stdout
        # ? Setup Custom Logging
        self.logger = logging.getLogger()
        if not self.logger.hasHandlers() or file_handler_path is not None:
            self._setup_logger(file_handler_path)

        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(
                    logging.Formatter(
                        "[%(asctime)s][%(levelname)s]\n%(message)s\n"
                    )
                )

    def _setup_logger(self, fp: Optional[str] = None) -> None:
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler())
        if fp is not None:
            self.logger.addHandler(logging.FileHandler(fp))
        else:
            self.logger.addHandler(
                logging.FileHandler(
                    os.path.join(self.output_directory, "bookkeeping.log")
                )
            )

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log a message.

        Args:
            message: Message to log.
            level: Logging level. Defaults to logging.INFO.
        """
        self.logger.log(level, message)

    def log_to_file(self, message: str, level: int = logging.INFO) -> None:
        """Log a message to the file handler.

        Args:
            message: Message to log.
            level: Logging level. Defaults to logging.INFO.
        """
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.emit(
                    logging.LogRecord(
                        self.logger.name,
                        level,
                        None,
                        None,
                        message,
                        None,
                        None,
                    )
                )

    def create_folder(self, folder_name: str, path: str = "") -> None:
        """Create a new folder in the output directory.

        Args:
            folder_name: Name of the folder to create.
            path: Subdirectory to create the folder in. Defaults to "".
        """
        os.makedirs(
            os.path.join(self.output_directory, path, folder_name),
            exist_ok=True,
        )

    def save_model_summary(
        self,
        model: torch.nn.Module,
        dataset: "AbstractDataset",
        filename: str,
    ) -> None:
        """Save a model summary to a file.

        Args:
            model: Model to summarize.
            dataset: Dataset to get the input size from.
            filename: Name of the file to save the summary to.
        """
        x = np.expand_dims(dataset[0][0], axis=0).shape
        with open(
            os.path.join(self.output_directory, filename),
            "w",
            encoding="utf-8",
        ) as f:
            sys.stdout = f
            s = summary(
                model=model,
                input_size=(x),
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "trainable",
                ],
                col_width=20,
                row_settings=["var_names"],
            )
            sys.stdout = self.original_stdout
        model_summary = {
            "total_mult_adds": s.total_mult_adds,
            "total_output_bytes": s.total_output_bytes,
            "total_params": s.total_params,
            "trainable_params": s.trainable_params,
            "total_param_bytes": s.total_param_bytes,
        }
        with open(
            os.path.join(
                self.output_directory, filename.replace(".txt", ".yaml")
            ),
            "w",
        ) as f:
            yaml.dump(model_summary, f)

    def save_state(
        self,
        obj: Union[
            torch.nn.Module,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler,
        ],
        filename: str,
        path: str = "",
    ) -> None:
        """Save the state of an object.

        Args:
            obj: Object to save the state of.
            filename: Name of the file to save the state to.
            path: Subdirectory to save the state to. Defaults to "".

        Raises:
            TypeError: If the object type is not supported.
        """
        p = os.path.join(self.output_directory, path, filename)
        _i = (
            torch.nn.Module,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler,
        )
        if not isinstance(obj, _i):
            raise TypeError(
                f"save_state of type {type(obj)} is not supported."
            )

        os.makedirs(os.path.join(self.output_directory, path), exist_ok=True)
        torch.save(obj.state_dict(), p)

    def load_state(
        self,
        obj: Union[
            torch.nn.Module,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler,
        ],
        filename: str,
        path: str = "",
    ) -> None:
        """Load the state of an object.

        Args:
            obj: Object to load the state into.
            filename: Name of the file to load the state from.
            path: Subdirectory to load the state from. Defaults to "".

        Raises:
            TypeError: If the object type is not supported.
            FileNotFoundError: If the file is not found.
        """
        p = os.path.join(self.output_directory, path, filename)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File {p} not found.")
        _i = (
            torch.nn.Module,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler,
        )
        if not isinstance(obj, _i):
            raise TypeError(
                f"load_state of type {type(obj)} is not supported."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            state_dict = torch.load(p, map_location="cpu", weights_only=True)
        obj.load_state_dict(state_dict)

    def save_audobject(
        self,
        obj: audobject.Object,
        filename: str,
        path: str = "",
    ) -> None:
        """Save an audobject.Object to disk.

        Args:
            obj: Object to save.
            filename: Name of the file to save the object to.
            path: Subdirectory to save the object to. Defaults to "".

        Raises:
            TypeError: If the object type is not supported.
        """
        if not isinstance(obj, audobject.Object):
            raise TypeError(
                f"save_audobject of type {type(obj)} is not supported."
            )
        os.makedirs(os.path.join(self.output_directory, path), exist_ok=True)
        obj.to_yaml(os.path.join(self.output_directory, path, filename))

    def save_results_dict(
        self,
        results_dict: Dict[str, float],
        filename: str,
        path: str = "",
    ) -> None:
        """Save a results dictionary to disk.

        Args:
            results_dict: Dictionary of metric names and values to save.
            filename: Name of the file to save the results to.
            path: Subdirectory to save the results to. Defaults to "".
        """
        os.makedirs(os.path.join(self.output_directory, path), exist_ok=True)
        with open(
            os.path.join(self.output_directory, path, filename),
            "w",
            encoding="utf-8",
        ) as f:
            yaml.dump(results_dict, f)

    def save_results_df(
        self, results_df: pd.DataFrame, filename: str, path: str = ""
    ) -> None:
        """Save a results DataFrame to disk.

        Args:
            results_df: DataFrame to save.
            filename: Name of the file to save the results to.
            path: Subdirectory to save the results to. Defaults to "".
        """
        os.makedirs(os.path.join(self.output_directory, path), exist_ok=True)
        results_df.to_csv(
            os.path.join(self.output_directory, path, filename), index=False
        )

    def save_results_np(
        self, results_np: np.ndarray, filename: str, path: str = ""
    ) -> None:
        """Save a results numpy array to disk.

        Args:
            results_np: Numpy array to save.
            filename: Name of the file to save the results to.
            path: Subdirectory to save the results to. Defaults to "".
        """
        os.makedirs(os.path.join(self.output_directory, path), exist_ok=True)
        np.save(
            os.path.join(self.output_directory, path, filename), results_np
        )

    def save_best_results(
        self,
        metrics: pd.DataFrame,
        filename: str,
        metric_fns: List[AbstractMetric],
        tracking_metric_fn: AbstractMetric,
        path: str = "",
    ) -> None:
        """Save the best results to disk.

        Args:
            metrics: DataFrame of metrics to save.
            filename: Name of the file to save the best results to.
            metric_fns: List of metric functions to get the best results from.
            tracking_metric_fn: Tracking metric function to get the best
                iteration from.
            path: Subdirectory to save the best results to. Defaults to "".
        """
        best_metrics = {}
        for m in metrics:
            if "loss" in m:
                best_metrics[f"{m}_min"] = float(metrics[m].min())

        for m in metric_fns:
            best_metrics[m.name] = m.get_best(metrics[m.name])

        best_metrics["best_iteration"] = tracking_metric_fn.get_best_pos(
            metrics[tracking_metric_fn.name]
        )
        os.makedirs(os.path.join(self.output_directory, path), exist_ok=True)
        with open(
            os.path.join(self.output_directory, path, filename),
            "w",
            encoding="utf-8",
        ) as f:
            yaml.dump(best_metrics, f)

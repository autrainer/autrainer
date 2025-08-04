from typing import Any, Callable, Dict, List, TypeVar

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad

from autrainer.core.utils import Bookkeeping
from autrainer.datasets import AbstractDataset


T = TypeVar("T")


class OutputsTracker:
    def __init__(
        self,
        export: bool,
        prefix: str,
        data: AbstractDataset,
        bookkeeping: Bookkeeping,
    ) -> None:
        """Tracker for model outputs, targets, losses, and predictions.

        Args:
            export: Whether to export the results.
            prefix: Prefix for the exported files.
            data: Instance of the dataset.
            bookkeeping: Instance of the bookkeeping class.
        """
        self._export = export
        self._prefix = prefix
        self._data = data
        self._bookkeeping = bookkeeping
        self.reset()

    def reset(self) -> None:
        """Reset the tracker. Clears all the stored data accumulated over a
        single iteration.
        """
        self._outputs = torch.zeros((0, self._data.output_dim), dtype=torch.float32)
        self._targets = torch.zeros(
            0, dtype=torch.long
        )  # automatic type promotion in case of float targets
        self._indices = torch.zeros(0, dtype=torch.long)
        self._losses = torch.zeros(0, dtype=torch.float32)
        self._predictions = None
        self._results_df = None

    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        loss: torch.Tensor,
        sample_idx: torch.Tensor,
    ) -> None:
        """Update the tracker with the current model outputs, targets, losses,
        and sample indices.

        Args:
            output: Detached model outputs.
            target: Targets.
            loss: Per-sample losses.
            sample_idx: Sample indices.
        """
        self._outputs = torch.cat([self._outputs, output.cpu()], dim=0)
        self._targets = torch.cat([self._targets, target.cpu()], dim=0)
        self._losses = torch.cat([self._losses, loss.cpu()], dim=0)
        self._indices = torch.cat([self._indices, sample_idx.cpu()], dim=0)

    def save(self, iteration_folder: str, reset: bool = True) -> None:
        """Save the tracked data to disk.

        Args:
            iteration_folder: Current iteration folder.
            reset: Whether to reset the tracker after saving the results.
                Defaults to True.
        """
        results = {
            "outputs": self._outputs.numpy(),
            "targets": self._targets.numpy(),
            "indices": self._indices.numpy(),
            "losses": self._losses.numpy(),
        }

        _probabilities = self._data.target_transform.probabilities_inference(
            self._outputs
        )
        self._predictions = self._data.target_transform.predict_inference(
            _probabilities
        )
        self._results_df = pd.DataFrame(index=results["indices"])
        self._results_df["predictions"] = self._predictions
        self._results_df["predictions"] = self._results_df["predictions"].apply(
            self._data.target_transform.decode
        )
        _probs_df = pd.DataFrame(
            index=results["indices"],
            data=(
                self._data.target_transform.probabilities_to_dict(p)
                for p in _probabilities
            ),
        )
        self._results_df = pd.concat([self._results_df, _probs_df], axis=1)

        if self._export:
            for key, value in results.items():
                self._bookkeeping.save_results_np(
                    value, f"{self._prefix}_{key}.npy", iteration_folder
                )
            self._bookkeeping.save_results_df(
                self._results_df.reset_index(),
                f"{self._prefix}_results.csv",
                iteration_folder,
            )

        if reset:
            self.reset()

    @staticmethod
    def check_saved(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(self: "OutputsTracker", *args: Any, **kwargs: Dict[str, Any]) -> T:
            if self._results_df is None:
                raise ValueError("Results not saved yet.")
            return func(self, *args, **kwargs)

        return wrapper

    @property
    @check_saved
    def outputs(self) -> np.ndarray:
        """Get the model outputs.

        Returns:
            Model outputs.
        """
        return self._outputs.numpy()

    @property
    @check_saved
    def targets(self) -> np.ndarray:
        """Get the targets.

        Returns:
            Targets.
        """
        return self._targets.numpy()

    @property
    @check_saved
    def indices(self) -> np.ndarray:
        """Get the sample indices.

        Returns:
            Sample indices.
        """
        return self._indices.numpy()

    @property
    @check_saved
    def losses(self) -> np.ndarray:
        """Get the per-sample losses.

        Returns:
            Per-sample losses.
        """
        return self._losses.numpy()

    @property
    @check_saved
    def predictions(self) -> np.ndarray:
        """Get the predictions.

        Returns:
            Predictions.
        """
        return np.array(self._predictions)

    @property
    @check_saved
    def results_df(self) -> pd.DataFrame:
        """Get the results as a DataFrame.

        Returns:
            Results as a DataFrame.
        """
        return self._results_df


class SequentialOutputsTracker(OutputsTracker):
    def reset(self) -> None:
        """Reset the tracker. Clears all the stored data accumulated over a
        single iteration.
        """
        self._outputs = []
        self._targets = []
        self._indices = torch.zeros(0, dtype=torch.long)
        self._losses = torch.zeros(0, dtype=torch.float32)
        self._masks = None
        self._predictions = None
        self._results_df = None

    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        loss: torch.Tensor,
        sample_idx: torch.Tensor,
    ) -> None:
        """Update the tracker with the current model outputs, targets, losses,
        and sample indices.

        Args:
            output: Detached model outputs.
            target: Targets.
            loss: Per-sample losses.
            sample_idx: Sample indices.
        """
        self._outputs += [output.cpu()]
        self._targets += [target.cpu()]
        self._losses = torch.cat([self._losses, loss.cpu()], dim=0)
        self._indices = torch.cat([self._indices, sample_idx.cpu()], dim=0)

    def save(self, iteration_folder: str, reset: bool = True) -> None:
        """Save the tracked data to disk.

        Args:
            iteration_folder: Current iteration folder.
            reset: Whether to reset the tracker after saving the results.
                Defaults to True.
        """
        # get length of longest sequence for padding & masking
        num_sequences = len(self._outputs)
        max_length = max([x.shape[0] for x in self._outputs])
        self._masks = torch.zeros(len(self._outputs), max_length)
        indices = []
        for index, x in enumerate(self._outputs):
            self._masks[index, : len(x)] = 1
            indices += [self._indices[index]] * len(x)
        self._outputs = torch.cat(
            [pad(x, [0, 0, 0, max_length], value=-100) for x in self._outputs]
        )
        self._targets = torch.cat(
            [pad(x, [0, 0, 0, max_length], value=-100) for x in self._targets]
        )
        results = {
            "outputs": self._outputs.numpy(),
            "targets": self._targets.numpy(),
            "indices": self._indices.numpy(),
            "losses": self._losses.numpy(),
        }

        _probabilities = self._data.target_transform.probabilities_inference(
            self._outputs
        )
        self._predictions = self._data.target_transform.predict_inference(
            _probabilities
        )
        print(self._outputs.shape)
        print(self._targets.shape)
        print(np.array(self._predictions).shape)
        print(
            np.array(self._predictions)
            .reshape(num_sequences * max_length, -1)
            .shape
        )
        exit()
        self._results_df = pd.DataFrame(index=indices)
        self._results_df["predictions"] = np.array(self._predictions).reshape(
            num_sequences * max_length, -1
        )
        self._results_df["predictions"] = self._results_df[
            "predictions"
        ].apply(self._data.target_transform.decode)
        self._results_df["masks"] = self._masks
        _probs_df = pd.DataFrame(
            index=indices,
            data=(
                self._data.target_transform.probabilities_to_dict(p)
                for p in _probabilities
            ),
        )
        self._results_df = pd.concat([self._results_df, _probs_df], axis=1)

        if self._export:
            for key, value in results.items():
                self._bookkeeping.save_results_np(
                    value, f"{self._prefix}_{key}.npy", iteration_folder
                )
            self._bookkeeping.save_results_df(
                self._results_df.reset_index(),
                f"{self._prefix}_results.csv",
                iteration_folder,
            )

        if reset:
            self.reset()

    def check_saved(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(self: "SequentialOutputsTracker", *args, **kwargs) -> T:
            if self._results_df is None:
                raise ValueError("Results not saved yet.")
            return func(self, *args, **kwargs)

        return wrapper

    @property
    @check_saved
    def masks(self) -> np.ndarray:
        """Get the masks.

        Returns:
            Masks.
        """
        return np.array(self._masks)


def init_trackers(
    exports: List[bool],
    prefixes: List[str],
    data: AbstractDataset,
    bookkeeping: Bookkeeping,
) -> List[OutputsTracker]:
    """Utility function to initialize multiple trackers at once.

    Args:
        exports: Whether to export the results.
        prefixes: Prefixes for the exported files.
        data: Instance of the dataset.
        bookkeeping: Instance of the bookkeeping class.

    Returns:
        List of initialized trackers.
    """
    trackers = []
    for export, prefix in zip(exports, prefixes, strict=False):
        trackers.append(
            OutputsTracker(
                export=export,
                prefix=prefix,
                data=data,
                bookkeeping=bookkeeping,
            )
        )

    return trackers

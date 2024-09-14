import copy
from typing import List

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch

from autrainer.core.utils import Bookkeeping
from autrainer.datasets import AbstractDataset


class OutputsTracker:
    def __init__(
        self,
        export: bool,
        prefix: str,
        data: AbstractDataset,
        criterion: DictConfig,
        bookkeeping: Bookkeeping = None,
    ) -> None:
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._export = export
        self._prefix = prefix
        self._data = data
        self._bookkeeping = bookkeeping
        self._tracker_criterion = copy.deepcopy(criterion).to(self._device)
        self._tracker_criterion.reduction = "none"
        self.reset()

    def reset(self) -> None:
        self._outputs = torch.zeros(
            (0, self._data.output_dim), dtype=torch.float32
        )
        self._targets = torch.zeros(0, dtype=torch.long)
        self._indices = torch.zeros(0, dtype=torch.long)
        self._losses = torch.zeros(0, dtype=torch.float32)
        self._predictions = None
        self._results_df = None

    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        sample_idx: torch.Tensor,
    ) -> None:
        self._outputs = torch.cat([self._outputs, output.cpu()], dim=0)
        self._targets = torch.cat([self._targets, target.cpu()], dim=0)
        self._indices = torch.cat([self._indices, sample_idx], dim=0)

        with torch.no_grad():
            loss = self._tracker_criterion(
                output.to(self._device), target.to(self._device)
            )
            self._losses = torch.cat([self._losses, loss.cpu()], dim=0)

    def save(self, iteration_folder: str, reset=True) -> None:
        results = {
            "outputs": self._outputs.numpy(),
            "targets": self._targets.numpy(),
            "indices": self._indices.numpy(),
            "losses": self._losses.numpy(),
        }

        self._predictions = self._data.target_transform.predict_batch(
            self._outputs
        )
        self._results_df = pd.DataFrame(index=results["indices"])
        self._results_df["predictions"] = self._predictions
        self._results_df["predictions"] = self._results_df[
            "predictions"
        ].apply(self._data.target_transform.decode)

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

    def check_saved(func):
        def wrapper(self, *args, **kwargs):
            if self._results_df is None:
                raise ValueError("Results not saved yet.")
            return func(self, *args, **kwargs)

        return wrapper

    @property
    @check_saved
    def outputs(self) -> np.ndarray:
        return self._outputs.numpy()

    @property
    @check_saved
    def targets(self) -> np.ndarray:
        return self._targets.numpy()

    @property
    @check_saved
    def indices(self) -> np.ndarray:
        return self._indices.numpy()

    @property
    @check_saved
    def losses(self) -> np.ndarray:
        return self._losses.numpy()

    @property
    @check_saved
    def predictions(self) -> np.ndarray:
        return self._predictions

    @property
    @check_saved
    def results_df(self) -> pd.DataFrame:
        return self._results_df


def init_trackers(
    exports: List[bool],
    prefixes: List[str],
    data: AbstractDataset,
    criterion: DictConfig,
    bookkeeping: Bookkeeping = None,
) -> OutputsTracker:
    trackers = []
    for export, prefix in zip(exports, prefixes):
        trackers.append(
            OutputsTracker(
                export=export,
                prefix=prefix,
                data=data,
                criterion=criterion,
                bookkeeping=bookkeeping,
            )
        )

    return trackers

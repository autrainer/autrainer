from functools import cached_property
import os
import shutil
from typing import Any, Dict, List

import pandas as pd

from autrainer.datasets import BaseClassificationDataset
from autrainer.datasets.utils import ZipDownloadManager


FILES = {"ESC-50.zip": "https://github.com/karoldvl/ESC-50/archive/master.zip"}


class ESC50(BaseClassificationDataset):
    def __init__(
        self,
        train_folds: List[int],
        dev_folds: List[int],
        test_folds: List[int],
        **kwargs: Dict[str, Any],  # kwargs only for simplicity in the tutorial
    ) -> None:
        self.train_folds = train_folds
        self.dev_folds = dev_folds
        self.test_folds = test_folds
        super().__init__(**kwargs)

    @cached_property
    def _load_metadata(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.path, "esc50.csv"))

    @cached_property
    def df_train(self) -> pd.DataFrame:
        meta = self._load_metadata
        return meta[meta["fold"].isin(self.train_folds)]

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        meta = self._load_metadata
        return meta[meta["fold"].isin(self.dev_folds)]

    @cached_property
    def df_test(self) -> pd.DataFrame:
        meta = self._load_metadata
        return meta[meta["fold"].isin(self.test_folds)]

    @staticmethod
    def download(path: str) -> None:
        if os.path.exists(os.path.join(path, "default")):
            return

        dl_manager = ZipDownloadManager(FILES, path)
        dl_manager.download(check_exist=["ESC-50.zip"])
        dl_manager.extract(check_exist=["ESC-50-master"])
        shutil.move(
            os.path.join(path, "ESC-50-master", "audio"),
            os.path.join(path, "default"),
        )
        shutil.move(
            os.path.join(path, "ESC-50-master", "meta", "esc50.csv"),
            path,
        )
        shutil.rmtree(os.path.join(path, "ESC-50-master"))

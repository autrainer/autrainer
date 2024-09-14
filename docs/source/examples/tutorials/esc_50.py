import os
import shutil

import pandas as pd

from autrainer.datasets import BaseClassificationDataset
from autrainer.datasets.utils import ZipDownloadManager


FILES = {"ESC-50.zip": "https://github.com/karoldvl/ESC-50/archive/master.zip"}


class ESC50(BaseClassificationDataset):
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

        meta_path = os.path.join(path, "ESC-50-master", "meta", "esc50.csv")
        meta = pd.read_csv(meta_path)

        # Simple split only for demonstration purposes
        meta[meta["fold"] < 4].to_csv(os.path.join(path, "train.csv"))
        meta[meta["fold"] == 4].to_csv(os.path.join(path, "dev.csv"))
        meta[meta["fold"] == 5].to_csv(os.path.join(path, "test.csv"))
        shutil.rmtree(os.path.join(path, "ESC-50-master"))

import os
import json
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, List


class AudioSet(Dataset):
    def __init__(
        self,
        split: str,
        data_path: str,
        features_path: str,
        features_subdir: str,
        file_type: str,
        label_map: str,
        transform: Optional[object] = None,
        logger: Optional[object] = None,
    ):
        self.split = split
        self.data_path = data_path
        self.features_path = features_path
        self.features_subdir = features_subdir
        self.file_type = file_type
        self.transform = transform
        self.label_map = self._load_label_map(label_map)
        self._log = logger or self._get_default_logger()
        self.df = self._load_df()

    def _get_default_logger(self):
        import logging

        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _load_label_map(self, label_map_path: str) -> dict:
        with open(label_map_path) as f:
            return json.load(f)

    def _load_df(self) -> pd.DataFrame:
        # Load metadata CSV
        csv_path = os.path.join(self.data_path, f"{self.split}.csv")
        df = pd.read_csv(csv_path)

        # Convert comma-separated label strings into lists
        df["labels"] = df["labels"].apply(lambda x: x.split(","))

        # Check if any clips are shorter than the expected sliding window duration
        min_required_duration = 10  # in seconds
        short_clips = []
        for i, row in df.iterrows():
            audio_path = os.path.join(
                self.features_path, self.features_subdir, row["filename"].replace(".wav", f".{self.file_type}")
            )
            if os.path.isfile(audio_path):
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    duration = waveform.shape[1] / sr
                    if duration < min_required_duration:
                        short_clips.append(row["filename"])
                except Exception as e:
                    self._log.warning(f"Error reading audio file {audio_path}: {e}")

        # Warn user if short clips are found
        if short_clips:
            self._log.warning(
                f"{len(short_clips)} files are shorter than the sliding window duration of {min_required_duration}s. "
                f"Examples: {short_clips[:5]}"
            )

        return df

    def map_to_classes(self, labels: List[str]) -> torch.Tensor:
        # Map label list to a multi-hot encoded tensor
        targets = torch.zeros(len(self.label_map), dtype=torch.float32)
        for label in labels:
            if label in self.label_map:
                targets[self.label_map[label]] = 1.0
        return targets

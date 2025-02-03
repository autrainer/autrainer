import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import audiofile
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from autrainer.transforms import SmartCompose

from .file_handlers import AbstractFileHandler, AudioFileHandler
from .target_transforms import AbstractTargetTransform


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        index_column: str,
        target_column: Union[str, List[str]],
        file_type: str,
        file_handler: AbstractFileHandler,
        df: pd.DataFrame,
        transform: Optional[SmartCompose] = None,
        target_transform: Optional[AbstractTargetTransform] = None,
    ):
        """Wrapper around torch.utils.data.Dataset.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            df: Dataframe containing the index and target column(s).
            transform: Transform to apply to the features. Defaults to None.
            target_transform: Target transform to apply to the target.
                Defaults to None.

        """
        self.path = path
        self.features_subdir = features_subdir
        self.index_column = index_column
        self.target_column = target_column
        self.file_type = file_type
        self.file_handler = file_handler
        self.df = df.copy()
        self.transform = transform
        self.target_transform = target_transform

        self.df[self.index_column] = self.df[self.index_column].apply(
            self._create_file_path
        )

    def _create_file_path(self, file: str) -> str:
        path = Path(self.path, self.features_subdir, file)
        path = path.with_suffix(f".{self.file_type}")
        return str(path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self,
        item: int,
    ) -> Tuple[torch.Tensor, Union[int, torch.Tensor], int]:
        """Get item from the dataset.

        Args:
            item: Index of the item.

        Returns:
            Tuple containing the data, target and item index.
        """
        index = self.df.index[item]
        item_path = self.df.loc[index, self.index_column]
        data = self.file_handler.load(item_path)
        target = self.df.loc[index, self.target_column]
        if isinstance(target, pd.Series):
            target = torch.Tensor(target.to_list())

        if self.transform is not None:
            data = self.transform(data, index=item).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target, item


class SegmentedDatasetWrapper(DatasetWrapper):
    def __init__(
        self,
        window_size: float = 0.25,
        hop_size: float = 0.25,
        min_event_length: float = 0.25,
        min_event_gap: float = 0.15,
        **kwargs,
    ):
        """Wrapper around torch.utils.data.Dataset for segmented audio files.
        If data is already windowed (has 'start' and 'end' columns), windowing parameters are ignored.

        Args:
            window_size: Size of the segmented window in seconds. Ignored if data is pre-windowed.
            hop_size: Hop size in seconds. Ignored if data is pre-windowed.
            min_event_length: Minimum duration of an event in seconds. Ignored if data is pre-windowed.
            min_event_gap: Minimum duration between events in seconds. Ignored if data is pre-windowed.
        """
        super().__init__(**kwargs)
        self.is_prewindowed = (
            "start" in self.df.columns and "end" in self.df.columns
        )
        if not self.is_prewindowed:
            if window_size is None:
                raise ValueError(
                    "window_size must be provided for non-windowed data"
                )
            self.window_size = window_size
            self.hop_size = hop_size
            self.min_event_length = min_event_length
            self.min_event_gap = min_event_gap
            self.df = self.convert_to_fixed_windows(self.df)

    def __getitem__(
        self,
        item: int,
    ) -> Tuple[torch.Tensor, Union[int, torch.Tensor], int]:
        """Get item from the dataset.

        Args:
            item: Index of the item.

        Returns:
            Tuple containing the data, target and item index.
        """

        index = self.df.index[item]
        item_path = self.df.loc[index, self.index_column]
        if isinstance(self.file_handler, AudioFileHandler):
            data = audiofile.read(
                item_path,
                offset=self.df.loc[index, "start"],
                duration=self.df.loc[index, "end"]
                - self.df.loc[index, "start"],
                always_2d=True,
            )[0]
            data = torch.from_numpy(data)
        else:
            data = self.file_handler.load(item_path)

        target = self.df.loc[index, self.target_column]
        if isinstance(target, str) and target.startswith("[["):
            target = torch.tensor(eval(target), dtype=torch.float32)
        elif isinstance(target, pd.Series):
            target = torch.tensor(float(target), dtype=torch.float32)
        if self.transform is not None:
            data = self.transform(data, index=item).float()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target, item

    @staticmethod
    def create_fixed_windows(
        df: pd.DataFrame,
        path: str,
        window_size: float = 0.25,
        min_event_length: float = 0.25,
        event_list: Optional[List[str]] = None,
        seq2seq: bool = False,
        max_duration: float = 10.0,
    ) -> pd.DataFrame:
        """Static version of convert_to_fixed_windows for use during download.

        Event window:\\
        filename       start  end   Cat, Dog, ...\\
        1123.wav,     0.0,   0.25,  0,1,0,0,0,1,0,0,0,0\\

        Concatenated/sequential window:\\
        filename       start  end    segment_events\\
        1123.wav,     0.0,   10.0,  [[0,1,0,0,0,1,0,0,0,0],  # segment 0 (0.0-0.25s)\\
                                     [0,1,0,0,0,1,0,0,0,0],  # segment 1 (0.25-0.5s)\\
                                     [0,1,0,0,0,1,0,0,0,0],  # segment 2 (0.5-0.75s)\\
                                     ...,\\
                                     [0,1,0,0,0,1,0,0,0,0]]  # segment 39 (9.75-10.0s)\\
        
        TODO Bounding box format:\\
        filename    event_label  t_start  t_end   f_low  f_high  confidence\\
        1123.wav    Dog         0.5      1.2     200    2000    0.95\\
        1123.wav    Cat         1.8      2.3     300    3000    0.87\\
            - https://github.com/merlresearch/sebbs\\
            - https://www.merl.com/publications/docs/TR2024-118.pdf\\

        TODO Overlap ratio for weak labels:\\
        filename       start  end   Cat, Dog, ...\\
        1123.wav,     0.0,   0.25,  0.12, 0.68   \\

        Args:
            df: DataFrame with columns [filename, onset, offset, event_label]
            path: Path to audio files
            window_size: Size of the windows in seconds
            min_event_length: Minimum duration of an event in seconds
            event_list: Optional list of expected event labels. If provided,
                validates that all events in df match this list.
            include_overlap_ratio: If True, includes overlap ratio for each event
            seq2seq: If True, returns a sequence-to-sequence format
            max_duration: Maximum duration of an audio file in seconds
                
        Returns:
            DataFrame with columns [filename, start, end] + event_labels
        """

        # Check presence of event labels
        event_labels = sorted(df["event_label"].unique())
        if event_list is not None:
            unknown_events = set(event_labels) - set(event_list)
            missing_events = set(event_list) - set(event_labels)
            if unknown_events:
                raise ValueError(
                    f"Unknown event labels found: {unknown_events}"
                )
            if missing_events:
                print(
                    f"Warning: Some event labels not present in data: {missing_events}"
                )
            event_labels = event_list

        windows = []
        for file in tqdm(
            df["filename"].unique(), desc="Processing files", unit="file"
        ):
            file_path = os.path.join(path, file)
            file_duration = audiofile.duration(file_path)
            if file_duration > max_duration:
                file_duration = max_duration
            file_events = df[df["filename"] == file]

            # Process segments s_i
            segment_events = []
            for start in tqdm(
                np.arange(0, file_duration, window_size),
                desc=f"Processing segments for {file}",
                unit="segment",
                leave=False,
                total=len(np.arange(0, file_duration, window_size)),
            ):
                end = min(start + window_size, file_duration)
                segment_vector = [0] * len(event_labels)
                window = {
                    "filename": file,
                    "start": 0.0 if seq2seq else start,
                    "end": file_duration if seq2seq else end,
                }

                # Check event presence a_1, ... a_N in s_i
                for _, event in file_events.iterrows():
                    overlap_start = max(start, event["onset"])
                    overlap_end = min(end, event["offset"])
                    overlap_duration = max(0, overlap_end - overlap_start)
                    if overlap_duration > (min_event_length * 0.5):
                        event_idx = event_labels.index(event["event_label"])
                        segment_vector[event_idx] = 1

                if seq2seq:
                    # Collect segments over time
                    segment_events.append(segment_vector)
                else:
                    # Annotate segment_window with event_labels
                    window.update(
                        {
                            label: segment_vector[idx]
                            for idx, label in enumerate(event_labels)
                        }
                    )
                    windows.append(window)

            if seq2seq:
                # Annotate window with segment_events
                window["segment_events"] = segment_events
                windows.append(window)

        return pd.DataFrame(windows)

    def convert_to_fixed_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.create_fixed_windows(
            df,
            path=self.path,
            window_size=self.window_size,
            min_event_length=self.min_event_length,
        )

    def get_bounding_box(self, spectrogram, threshold=0.5):
        """Get bounding bo1 x for a spectrogram.

        https://www.merl.com/publications/docs/TR2024-118.pdf
        https://github.com/merlresearch/sebbs

        Args:
            spectrogram (_type_): _description_
            threshold (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """
        # Time bounds
        energy_over_time = np.mean(spectrogram, axis=0)
        time_mask = energy_over_time > threshold
        t_start = np.where(time_mask)[0][0]
        t_end = np.where(time_mask)[0][-1]

        # Frequency bounds
        energy_over_freq = np.mean(spectrogram, axis=1)
        freq_mask = energy_over_freq > threshold
        f_low = np.where(freq_mask)[0][0]
        f_high = np.where(freq_mask)[0][-1]

        return t_start, t_end, f_low, f_high

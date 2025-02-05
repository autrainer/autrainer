from pathlib import Path
from typing import List, Optional, Tuple, Union

import audiofile
import pandas as pd
import torch

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
        file_handler: AbstractFileHandler,
        df: pd.DataFrame,
        file_type: Optional[str] = None,
        transform: Optional[SmartCompose] = None,
        target_transform: Optional[AbstractTargetTransform] = None,
    ):
        """Wrapper around torch.utils.data.Dataset.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_handler: File handler to load the data.
            df: Dataframe containing the index and target column(s).
            file_type: File type of the features. If `None`,
                will not enforce a file_type. This can be useful
                in case the dataset contains audio files
                with different formats. Defaults to `None`.
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
        if self.file_type is not None:
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
        if isinstance(target, pd.Series):
            target = torch.Tensor(target.to_list())

        if self.transform is not None:
            data = self.transform(data, index=item).float()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target, item

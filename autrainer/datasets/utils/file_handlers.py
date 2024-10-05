from abc import ABC, abstractmethod
from typing import Optional

import audiofile
import audobject
import numpy as np
import torch
import torchaudio
import torchvision


class AbstractFileHandler(ABC, audobject.Object):
    def __init__(self) -> None:
        """Abstract file handler for loading files in the dataset and saving
        files during preprocessing.

        Serves as the base for creating custom file handlers that handle
        loading and saving of different file types.
        """

    def __call__(self, file: str) -> torch.Tensor:
        """Load a file from a path.

        Args:
            file: Path to file.

        Returns:
            Loaded file.
        """
        return self.load(file)

    @abstractmethod
    def load(self, file: str) -> torch.Tensor:
        """Load a file from a path.

        Args:
            file: Path to file.

        Returns:
            Loaded file.
        """

    @abstractmethod
    def save(self, file: str, data: torch.Tensor) -> None:
        """Save a file to a path.

        Args:
            file: Path to file.
            data: Data to save.
        """


class IdentityFileHandler(AbstractFileHandler):
    def __init__(self) -> None:
        """Identity file handler serving as a no-op. Both load and save methods
        return None.
        """

    def load(self, file: str) -> None:
        """Identity operation.

        Args:
            file: Path to file.
        """
        return file

    def save(self, file: str, data: torch.Tensor) -> None:
        """Identity operation.

        Args:
            file: Path to file.
            data: Data to save.
        """


class ImageFileHandler(AbstractFileHandler):
    def __init__(self) -> None:
        """Image file handler for loading and saving with torchvision.
        Torchvision supports the PNG, JPEG, and GIF image formats for loading
        and saving images.
        """

    def load(self, file: str) -> torch.Tensor:
        """Load an image from a file as a uint8 tensor in the range [0, 255].

        Args:
            file: Path to image file.

        Returns:
            Uint8 image tensor.
        """
        return torchvision.io.read_image(file)

    def save(self, file: str, data: torch.Tensor) -> None:
        """Save an image tensor to a file.

        If the tensor is of type uint8, it is assumed to be in the range
        [0, 255] and divided by 255 before saving.

        Args:
            file: Path to image file.
            data: Image tensor to save.
        """
        if data.dtype == torch.uint8:
            data = data / 255
        torchvision.utils.save_image(data, file)


class NumpyFileHandler(AbstractFileHandler):
    def __init__(self) -> None:
        """Numpy file handler for loading and saving numpy arrays."""

    def load(self, file: str) -> torch.Tensor:
        """Load a numpy array from a file.

        Args:
            file: Path to numpy file.

        Returns:
            Numpy array as a tensor.
        """
        return torch.from_numpy(np.load(file))

    def save(self, file: str, data: torch.Tensor) -> None:
        """Save a tensor to a numpy file.

        Args:
            file: Path to numpy file.
            data: Tensor to save.
        """
        np.save(file, data.numpy())


class AudioFileHandler(AbstractFileHandler):
    def __init__(
        self,
        target_sample_rate: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Audio file handler with optional resampling.

        Args:
            target_sample_rate: Target sample rate to resample audio files to
                during loading. Has to be specified to save audio files.
                If None, audio files are loaded with their
                original sample rate. Defaults to None.
            **kwargs: Additional keyword arguments passed to
                torchaudio.transforms.Resample.
        """
        # ? audobject passes _object_root_ to the object, which is not a valid
        # ? argument for the file handler.
        kwargs.pop("_object_root_", None)
        self.target_sample_rate = target_sample_rate
        self.kwargs = kwargs

    def load(self, file: str) -> torch.Tensor:
        """Load an audio file and resample it if a target sample rate is
        specified.

        Args:
            file: Path to audio file.

        Returns:
            Loaded audio file as a tensor.
        """
        x, sr = audiofile.read(file, always_2d=True)
        if (
            self.target_sample_rate is not None
            and sr != self.target_sample_rate
        ):
            resample = torchaudio.transforms.Resample(
                sr,
                self.target_sample_rate,
                **self.kwargs,
            )
            x = resample(torch.from_numpy(x)).numpy()
        return torch.from_numpy(x)

    def save(self, file: str, data: torch.Tensor) -> None:
        """Save an audio tensor to a file.

        Args:
            file: Path to audio file.
            data: Audio data to save.

        Raises:
            ValueError: If target sample rate is not specified.
        """
        if self.target_sample_rate is None:
            raise ValueError(
                "Target sample rate has to be specified to save audio files."
            )
        audiofile.write(file, data.numpy(), self.target_sample_rate)

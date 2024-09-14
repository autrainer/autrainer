import torch

from autrainer.datasets.utils import AbstractFileHandler


class TorchFileHandler(AbstractFileHandler):
    def load(self, file: str) -> torch.Tensor:
        return torch.load(file)

    def save(self, file: str, data: torch.Tensor) -> None:
        torch.save(data, file)

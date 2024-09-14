from typing import List

import torch

from autrainer.models import AbstractModel


class SpectrogramCNN(AbstractModel):
    def __init__(self, output_dim: int, hidden_dims: List[int]) -> None:
        """Spectrogram CNN model with a variable number of hidden CNN layers.

        Args:
            output_dim: Output dimension of the model.
            hidden_dims: List of hidden dimensions for the CNN layers.
        """
        super().__init__(output_dim)
        self.hidden_dims = hidden_dims
        layers = []
        input_dim = 1
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    torch.nn.Conv2d(input_dim, hidden_dim, (3, 3), 1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((2, 2)),
                ]
            )
            input_dim = hidden_dim
        layers.extend(
            [
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
            ]
        )
        self.backbone = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(self.hidden_dims[-1], output_dim)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embeddings(x))

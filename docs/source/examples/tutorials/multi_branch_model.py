import torch

from autrainer.models import AbstractModel


class ToyMultiBranchModel(AbstractModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__(output_dim, None)  # no transfer learning
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim * 2, self.output_dim)

    def embeddings(
        self, features: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        return torch.concat(
            [self.linear1(features), self.linear2(meta)], axis=1
        )

    def forward(
        self, features: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        return self.out(self.embeddings(features=features, meta=meta))

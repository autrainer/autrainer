import torch

from autrainer.criterions import BalancedCrossEntropyLoss
from autrainer.datasets import ToyDataset
from autrainer.models import FFNN
from autrainer.training import Trainer


if __name__ == "__main__":
    dataset = ToyDataset(
        task="classification",
        size=1000,
        num_targets=10,
        feature_shape=64,
        dev_split=0.2,
        test_split=0.2,
        seed=1,
        metrics=[
            "autrainer.metrics.Accuracy",
            "autrainer.metrics.UAR",
            "autrainer.metrics.F1",
        ],
        tracking_metric="autrainer.metrics.Accuracy",
    )
    model = FFNN(output_dim=10, input_size=64, hidden_size=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        batch_size=32,
        output_directory="results-api",
        criterion=BalancedCrossEntropyLoss(reduction="none"),
        iterations=5,
    )
    trainer.train()

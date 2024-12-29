from typing import Any, Type

import pandas as pd
import pytest
import torch

from autrainer.criterions import (
    BalancedCrossEntropyLoss,
    CrossEntropyLoss,
    MSELoss,
)


class TestBalancedCrossEntropyLoss:
    def test_setup(self) -> None:
        class MockDataset:
            target_column = "target"
            df_train = pd.DataFrame({"target": [0, 1, 1, 2, 2, 2]})

            @staticmethod
            def target_transform(x: Any) -> Any:
                return x

        criterion = BalancedCrossEntropyLoss()
        criterion.setup(MockDataset())
        assert (
            criterion.weight is not None
        ), "Should have calculated the weights"
        test_weights = torch.tensor([1 / 1, 1 / 2, 1 / 3], dtype=torch.float32)
        test_weights = test_weights * len(test_weights) / test_weights.sum()
        assert torch.allclose(
            criterion.weight, test_weights
        ), "Should have calculated frequency-based weights"


class TestCrossEntropyLoss:
    @pytest.mark.parametrize(
        "cls",
        [
            CrossEntropyLoss,
            CrossEntropyLoss,
            BalancedCrossEntropyLoss,
            BalancedCrossEntropyLoss,
        ],
    )
    def test_forward_dtype(
        self,
        cls: Type[CrossEntropyLoss],
    ) -> None:
        x = torch.rand(10, 5)
        y = torch.randint(0, 5, (10,), dtype=torch.float32)
        loss = cls()(x, y)
        assert isinstance(loss, torch.Tensor), "Should return a torch.Tensor"


class TestMSELoss:
    def test_forward_dtype(self) -> None:
        x = torch.rand(10, 5)
        y = torch.rand(10, 5)
        loss = MSELoss()(x, y)
        assert isinstance(loss, torch.Tensor), "Should return a torch.Tensor"

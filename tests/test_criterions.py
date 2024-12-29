from typing import Any, Type

import pandas as pd
import pytest
import torch

from autrainer.criterions import (
    BalancedBCEWithLogitsLoss,
    BalancedCrossEntropyLoss,
    CrossEntropyLoss,
    MSELoss,
)


class TestBalancedBCEWithLogitsLoss:
    def _mock_criterion_setup(self) -> BalancedBCEWithLogitsLoss:
        class MockDataset:
            class MockTargetTransform:
                labels = ["target1", "target2", "target3"]

                def __call__(self, x: Any) -> Any:
                    return torch.tensor(x)

            target_column = ["target1", "target2", "target3"]
            df_train = pd.DataFrame(
                {
                    "target1": [0, 1, 0, 1, 0],
                    "target2": [1, 0, 1, 0, 1],
                    "target3": [0, 0, 1, 1, 1],
                }
            )
            target_transform = MockTargetTransform()

        criterion = BalancedBCEWithLogitsLoss()
        criterion.setup(MockDataset())
        return criterion

    def test_setup(self) -> None:
        criterion = self._mock_criterion_setup()
        assert (
            criterion.weight is not None
        ), "Should have calculated the weights"
        test_weights = torch.tensor([1 / 2, 1 / 3, 1 / 3], dtype=torch.float32)
        test_weights = test_weights * len(test_weights) / test_weights.sum()
        assert torch.allclose(
            criterion.weight, test_weights
        ), "Should have calculated frequency-based weights"

    def test_forward_dtype(self) -> None:
        criterion = self._mock_criterion_setup()
        x = torch.rand(10, 3)
        y = torch.randint(0, 2, (10, 3), dtype=torch.long)
        loss = criterion(x, y)
        assert isinstance(loss, torch.Tensor), "Should return a torch.Tensor"


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
            BalancedCrossEntropyLoss,
        ],
    )
    def test_forward_dtype(
        self,
        cls: Type[CrossEntropyLoss],
    ) -> None:
        x = torch.rand(10, 5)
        y = torch.randint(0, 5, (10,), dtype=torch.long)
        loss = cls()(x, y)
        assert isinstance(loss, torch.Tensor), "Should return a torch.Tensor"


class TestMSELoss:
    def test_forward_dtype(self) -> None:
        x = torch.rand(10, 5)
        y = torch.rand(10, 5)
        loss = MSELoss()(x, y)
        assert isinstance(loss, torch.Tensor), "Should return a torch.Tensor"

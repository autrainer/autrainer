from typing import Any

import pandas as pd
import pytest
import torch

from autrainer.criterions import (
    BalancedBCEWithLogitsLoss,
    BalancedCrossEntropyLoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    MSELoss,
    WeightedBCEWithLogitsLoss,
    WeightedCrossEntropyLoss,
    WeightedMSELoss,
)


class MockCTargetTransform:
    def __init__(self) -> None:
        self.labels = ["target1", "target2", "target3", "target4", "target5"]

    def __len__(self) -> int:
        return len(self.labels)

    def __call__(self, x: Any) -> Any:
        return x


class MockClassificationDataset:
    def __init__(self) -> None:
        self.target_column = "target"
        self.df_train = pd.DataFrame({"target": [0, 1, 1, 2, 2, 2, 3, 4]})
        self.task = "classification"
        self.target_transform = MockCTargetTransform()


class MockMLCTargetTransform:
    def __init__(self) -> None:
        self.labels = ["target1", "target2", "target3", "target4", "target5"]

    def __len__(self) -> int:
        return len(self.labels)

    def __call__(self, x: Any) -> Any:
        return torch.tensor(x)


class MockMLClassificationDataset:
    def __init__(self) -> None:
        self.target_column = [
            "target1",
            "target2",
            "target3",
            "target4",
            "target5",
        ]
        self.df_train = pd.DataFrame(
            {
                "target1": [0, 1, 0, 1, 0],
                "target2": [1, 0, 1, 0, 1],
                "target3": [0, 0, 1, 1, 1],
                "target4": [1, 1, 1, 1, 1],
                "target5": [1, 0, 1, 1, 1],
            }
        )
        self.task = "ml-classification"
        self.target_transform = MockMLCTargetTransform()


class MockMTRTargetTransform:
    def __init__(self) -> None:
        self.target = ["target1", "target2", "target3", "target4", "target5"]

    def __len__(self) -> int:
        return len(self.target)

    def __call__(self, x: Any) -> Any:
        return torch.tensor(x)


class MockMTRegressionDataset(MockMLClassificationDataset):
    def __init__(self) -> None:
        super().__init__()
        self.task = "mt-regression"
        self.target_transform = MockMTRTargetTransform()


class TestBalancedCrossEntropyLoss:
    def _mock_criterion_setup(
        self,
    ) -> tuple[BalancedCrossEntropyLoss, torch.Tensor]:
        criterion = BalancedCrossEntropyLoss()
        criterion.setup(MockClassificationDataset())  # ty: ignore[invalid-argument-type]
        weights = torch.tensor(
            [1 / 1, 1 / 2, 1 / 3, 1, 1],
            dtype=torch.float32,
        )
        weights = weights * len(weights) / weights.sum()
        return criterion, weights

    def test_invalid_frequency_setup(self) -> None:
        criterion = BalancedCrossEntropyLoss()
        dataset = MockClassificationDataset()
        dataset.df_train = pd.DataFrame({"target": [0, 1, 1, 2, 2, 2, 3, 0]})
        with pytest.raises(ValueError, match="weighting requires a non-zero frequency"):
            criterion.setup(dataset)  # ty: ignore[invalid-argument-type]

    def test_setup(self) -> None:
        criterion, weights = self._mock_criterion_setup()
        assert criterion.weight is not None, "Should have calculated the weights"

        assert torch.allclose(criterion.weight, weights), (
            "Should have calculated frequency-based weights"
        )

    def test_forward_dtype(self) -> None:
        criterion, _ = self._mock_criterion_setup()
        x = torch.rand(10, 5)
        y = torch.randint(0, 5, (10,), dtype=torch.long)
        loss = criterion(x, y)
        assert isinstance(loss, torch.Tensor), "Should return a torch.Tensor"


class TestWeightedCrossEntropyLoss(TestBalancedCrossEntropyLoss):
    def _mock_criterion_setup(
        self,
    ) -> tuple[WeightedCrossEntropyLoss, torch.Tensor]:
        criterion = WeightedCrossEntropyLoss(
            class_weights={
                "target1": 1,
                "target2": 2,
                "target3": 3,
                "target4": 5,
                "target5": 4,
            }
        )
        criterion.setup(MockClassificationDataset())  # ty: ignore[invalid-argument-type]
        weights = torch.tensor([1, 2, 3, 5, 4], dtype=torch.float32)
        weights = weights * len(weights) / weights.sum()
        return criterion, weights

    def test_invalid_frequency_setup(self) -> None:
        criterion = WeightedCrossEntropyLoss(
            class_weights={
                "target1": 1,
                "target2": 2,
                "target3": 3,
                "target4": 5,
                "target5": 0,
            }
        )
        with pytest.raises(ValueError, match="weighting requires a non-zero frequency"):
            criterion.setup(MockClassificationDataset())  # ty: ignore[invalid-argument-type]

    def test_missing_target_weight(self) -> None:
        criterion = WeightedCrossEntropyLoss(class_weights={"target1": 1})
        with pytest.raises(ValueError, match="Missing class weight for label"):
            criterion.setup(MockClassificationDataset())  # ty: ignore[invalid-argument-type]


class TestBalancedBCEWithLogitsLoss(TestBalancedCrossEntropyLoss):
    def _mock_criterion_setup(self) -> tuple[BalancedBCEWithLogitsLoss, torch.Tensor]:  # ty: ignore[invalid-method-override]
        criterion = BalancedBCEWithLogitsLoss()
        criterion.setup(MockMLClassificationDataset())  # ty: ignore[invalid-argument-type]
        weights = torch.tensor(
            [1 / 2, 1 / 3, 1 / 3, 1 / 5, 1 / 4],
            dtype=torch.float32,
        )
        weights = weights * len(weights) / weights.sum()
        return criterion, weights

    def test_setup(self) -> None:
        criterion, weights = self._mock_criterion_setup()
        assert criterion.weights_buffer is not None, (
            "Should have calculated the weights"
        )

        assert torch.allclose(criterion.weights_buffer, weights), (
            "Should have calculated frequency-based weights"
        )

    def test_invalid_frequency_setup(self) -> None:
        criterion = BalancedBCEWithLogitsLoss()
        dataset = MockMLClassificationDataset()
        dataset.target_transform.labels += ["target6"]
        with pytest.raises(ValueError, match="6 columns passed, passed data had 5"):
            criterion.setup(dataset)  # ty: ignore[invalid-argument-type]

    def test_forward_dtype(self) -> None:
        criterion, _ = self._mock_criterion_setup()
        x = torch.rand(10, 5)
        y = torch.randint(0, 2, (10, 5), dtype=torch.long)
        loss = criterion(x, y)
        assert isinstance(loss, torch.Tensor), "Should return a torch.Tensor"


class TestWeightedBCEWithLogitsLoss(TestBalancedBCEWithLogitsLoss):
    def _mock_criterion_setup(
        self,
    ) -> tuple[WeightedBCEWithLogitsLoss, torch.Tensor]:
        criterion = WeightedBCEWithLogitsLoss(
            class_weights={
                "target1": 1,
                "target2": 2,
                "target3": 3,
                "target4": 5,
                "target5": 4,
            }
        )
        criterion.setup(MockMLClassificationDataset())  # ty: ignore[invalid-argument-type]
        weights = torch.tensor([1, 2, 3, 5, 4], dtype=torch.float32)
        weights = weights * len(weights) / weights.sum()
        return criterion, weights

    def test_invalid_frequency_setup(self) -> None:
        criterion = WeightedBCEWithLogitsLoss(
            class_weights={
                "target1": 1,
                "target2": 2,
                "target3": 3,
                "target4": 5,
                "target5": 0,
            }
        )
        with pytest.raises(ValueError, match="weighting requires a non-zero frequency"):
            criterion.setup(MockMLClassificationDataset())  # ty: ignore[invalid-argument-type]

    def test_missing_target_weight(self) -> None:
        criterion = WeightedBCEWithLogitsLoss(class_weights={"target1": 1})
        with pytest.raises(ValueError, match="Missing class weight for label"):
            criterion.setup(MockMLClassificationDataset())  # ty: ignore[invalid-argument-type]


class TestWeightedMSELoss(TestBalancedCrossEntropyLoss):
    def _mock_criterion_setup(self) -> tuple[WeightedMSELoss, torch.Tensor]:  # ty: ignore[invalid-method-override]
        criterion = WeightedMSELoss(
            target_weights={
                "target1": 1,
                "target2": 2,
                "target3": 3,
                "target4": 5,
                "target5": 4,
            }
        )
        criterion.setup(MockMTRegressionDataset())  # ty: ignore[invalid-argument-type]
        weights = torch.tensor([1, 2, 3, 5, 4], dtype=torch.float32)
        weights = weights * len(weights) / weights.sum()
        return criterion, weights

    def test_setup(self) -> None:
        criterion, weights = self._mock_criterion_setup()
        assert criterion.weights_buffer is not None, (
            "Should have calculated the weights"
        )

        assert torch.allclose(criterion.weights_buffer, weights), (
            "Should have calculated frequency-based weights"
        )

    def test_invalid_task(self) -> None:
        criterion = WeightedMSELoss(target_weights=None)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="multi-target regression"):
            criterion.setup(MockClassificationDataset())  # ty: ignore[invalid-argument-type]

    def test_invalid_frequency_setup(self) -> None:
        criterion = WeightedMSELoss(
            target_weights={
                "target1": 1,
                "target2": 2,
                "target3": 3,
                "target4": 5,
                "target5": 0,
            }
        )
        with pytest.raises(ValueError, match="weighting requires a non-zero frequency"):
            criterion.setup(MockMTRegressionDataset())  # ty: ignore[invalid-argument-type]

    def test_missing_target_weight(self) -> None:
        criterion = WeightedMSELoss(target_weights={"target1": 1})
        with pytest.raises(ValueError, match="Missing target weight for target"):
            criterion.setup(MockMTRegressionDataset())  # ty: ignore[invalid-argument-type]

    def test_forward_dtype(self) -> None:
        criterion, _ = self._mock_criterion_setup()
        x = torch.rand(10, 5)
        y = torch.rand(10, 5)
        loss = criterion(x, y)
        assert isinstance(loss, torch.Tensor), "Should return a torch.Tensor"


class TestLossForward:
    @pytest.mark.parametrize(
        ("cls", "y"),
        [
            (CrossEntropyLoss, torch.randint(0, 5, (10,), dtype=torch.long)),
            (
                BalancedCrossEntropyLoss,
                torch.randint(0, 5, (10,), dtype=torch.long),
            ),
            (
                BCEWithLogitsLoss,
                torch.randint(0, 2, (10, 5), dtype=torch.long),
            ),
            (MSELoss, torch.rand(10, 5)),
        ],
    )
    def test_forward_dtype(self, cls: type[CrossEntropyLoss], y: torch.Tensor) -> None:
        x = torch.rand(10, 5)
        loss = cls()(x, y)
        assert isinstance(loss, torch.Tensor), "Should return a torch.Tensor"

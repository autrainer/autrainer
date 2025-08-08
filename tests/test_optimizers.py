from copy import deepcopy

import pytest
import torch

from autrainer.core.structs import DataBatch
from autrainer.criterions import CrossEntropyLoss
from autrainer.models import FFNN
from autrainer.optimizers import SAM


class TestSAM:
    @classmethod
    def setup_class(cls) -> None:
        cls.model = FFNN(10, 64, 64)
        cls.base_optimizer = "torch.optim.SGD"

    def test_rho_invalid(self) -> None:
        with pytest.raises(ValueError):
            SAM(
                params=self.model.parameters(),
                base_optimizer=self.base_optimizer,
                rho=-0.01,
                lr=0.01,
            )

    def test_custom_step(self) -> None:
        original_state_dict = deepcopy(self.model.state_dict())
        self.model.train()
        data = DataBatch(
            torch.randn(10, 64),
            torch.randint(0, 10, (10,)),
            torch.tensor(list(range(10))),
        )
        criterion = CrossEntropyLoss()
        optimizer = SAM(
            params=self.model.parameters(),
            base_optimizer=self.base_optimizer,
            lr=0.01,
        )
        optimizer.custom_step(
            model=self.model,
            data=data,
            criterion=criterion,
            probabilities_fn=lambda x: x,
        )

        assert str(original_state_dict) != str(self.model.state_dict()), (
            "Should not be equal"
        )

    def test_missing_closure(self) -> None:
        optimizer = SAM(
            params=self.model.parameters(),
            base_optimizer=self.base_optimizer,
            lr=0.01,
        )
        with pytest.raises(AssertionError):
            optimizer.step()

    def test_load_state_dict(self) -> None:
        optimizer = SAM(
            params=self.model.parameters(),
            base_optimizer=self.base_optimizer,
            lr=0.01,
        )
        state_dict = optimizer.state_dict()
        optimizer = SAM(
            params=FFNN(10, 64, 64).parameters(),
            base_optimizer=self.base_optimizer,
            lr=0.01,
        )
        optimizer.load_state_dict(state_dict)
        assert str(optimizer.state_dict()) == str(state_dict), (
            "Should be equal"
        )

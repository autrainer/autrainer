from typing import List

import numpy as np
import numpy.testing
import pytest
import torch
import torch.testing
from utils import BaseIndividualTempDir

from autrainer.core.utils import Bookkeeping
from autrainer.datasets import ToyDataset
from autrainer.training import OutputsTracker, SequentialOutputsTracker


TOY_DATASET = ToyDataset(
    task="classification",
    size=100,
    num_targets=10,
    feature_shape=10,
    dev_split=0.3,
    test_split=0.3,
    seed=1,
    metrics=["autrainer.metrics.Accuracy"],
    tracking_metric="autrainer.metrics.Accuracy",
)


class TestOutputsTracker(BaseIndividualTempDir):
    def test_init(self) -> None:
        tracker = OutputsTracker(
            export=True,
            prefix="foo",
            data=TOY_DATASET,
            bookkeeping=Bookkeeping(self.temp_dir),
        )
        print(tracker)

    @pytest.mark.parametrize(
        "dims",
        [(1, 10), (2, 10), (2, 2, 10)],
    )
    def test_update(self, dims: List[int]) -> None:
        tracker = OutputsTracker(
            export=True,
            prefix="foo",
            data=TOY_DATASET,
            bookkeeping=Bookkeeping(self.temp_dir),
        )
        x = torch.rand(dims)
        tracker.update(x, x, torch.Tensor([0.1]), torch.IntTensor([1]))
        tracker.update(x, x, torch.Tensor([0.1]), torch.IntTensor([1]))
        tracker.save("foo")

        np.testing.assert_array_almost_equal(
            torch.cat((x, x)).numpy(), tracker._outputs, 5
        )


if __name__ == "__main__":
    TOY_DATASET = ToyDataset(
        task="classification",
        size=100,
        num_targets=10,
        feature_shape=10,
        dev_split=0.3,
        test_split=0.3,
        seed=1,
        metrics=["autrainer.metrics.Accuracy"],
        tracking_metric="autrainer.metrics.Accuracy",
    )
    tracker = SequentialOutputsTracker(
        export=True,
        prefix="foo",
        data=TOY_DATASET,
        bookkeeping=Bookkeeping("foo"),
    )
    x = torch.rand((3, 2, 10))
    y = torch.rand((5, 4, 10))
    tracker.update(
        x,
        x,
        torch.Tensor([0.1]),
        torch.IntTensor([0, 1, 2]),
        mask=torch.Tensor([[1, 1], [1, 1], [1, 0]]),
    )
    tracker.update(y, y, torch.Tensor([0.1]), torch.IntTensor([3, 4, 5, 6, 7]))
    tracker.save("bar", reset=False)

    np.testing.assert_array_almost_equal(
        torch.cat((torch.cat((x, torch.ones(3, 2, 10) * -100), dim=1), y)).numpy(),
        tracker._outputs,
    )

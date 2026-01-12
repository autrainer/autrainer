from typing import List

import numpy as np
import numpy.testing
import pytest
import torch
import torch.testing
from .utils import BaseIndividualTempDir

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
        "dims,tracker_class",
        [
            ((1, 10), OutputsTracker),
            ((2, 10), OutputsTracker),
            ((3, 2, 10), SequentialOutputsTracker),
        ],
    )
    def test_update(self, dims: List[int], tracker_class) -> None:
        tracker = tracker_class(
            export=True,
            prefix="foo",
            data=TOY_DATASET,
            bookkeeping=Bookkeeping(self.temp_dir)
        )
        x = torch.rand(dims)
        tracker.update(x, x, torch.Tensor([0.1]), torch.randint(high=10, size=(dims[0], 1)).squeeze(1))
        tracker.update(x, x, torch.Tensor([0.1]), torch.randint(high=10, size=(dims[0], 1)).squeeze(1))
        tracker.save("foo", reset=False)

        np.testing.assert_array_almost_equal(
            torch.cat((x, x)).numpy(), tracker._outputs
        )

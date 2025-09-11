import logging
from typing import List, Type

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from autrainer.metrics import (
    CCC,
    F1,
    MAE,
    MSE,
    PCC,
    UAR,
    AbstractMetric,
    Accuracy,
    MLAccuracy,
    MLF1Macro,
    MLF1Micro,
    MLF1Weighted,
)


class TestAllMetrics:
    def _test_metric(self, m: AbstractMetric) -> None:
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 1, 1, 1])
        score = m(x, y)
        assert isinstance(score, float), "Score should be a float."

    def _test_metric_invalid(self, m: AbstractMetric) -> None:
        x = np.array([0, 1, 0, np.nan])
        y = np.array([0, 1, 0, 1])
        assert not np.isnan(m(x, y)), "Score should not be nan."

    def _test_starting_metric(self, m: AbstractMetric) -> None:
        assert m.suffix in ["min", "max"], "Suffix should be min or max."
        if m.suffix == "max":
            assert m.starting_metric < 0, "Starting metric should be negative."
        else:
            assert m.starting_metric > 0, "Starting metric should be positive."

    def _test_comparisons(self, m: AbstractMetric) -> None:
        for x in [pd.Series([1, 2, 3, 4]), np.array([1, 2, 3, 4])]:
            if m.suffix == "max":
                assert m.compare(5, 4), "5 should be greater than 4."
                assert not m.compare(4, 5), "4 should not be greater than 5."
                assert m.get_best(x) == 4, "4 should be the best."
                assert m.get_best_pos(x) == 3, "4 should be at position 3."
            else:
                assert m.compare(4, 5), "4 should be less than 5."
                assert not m.compare(5, 4), "5 should not be less than 4."
                assert m.get_best(x) == 1, "1 should be the best."
                assert m.get_best_pos(x) == 0, "1 should be at position 0."

    def _test_result(
        self, m: AbstractMetric, truth: List, pred: List, res: float
    ) -> None:
        np.testing.assert_almost_equal(m(truth, pred), res, 2)

    @pytest.mark.parametrize("cls", [Accuracy, F1, UAR, CCC, MAE, MSE, PCC])
    def test_classification_regression_metrics(self, cls: Type[AbstractMetric]) -> None:
        self._test_metric(cls())
        self._test_metric_invalid(cls())
        self._test_starting_metric(cls())
        self._test_comparisons(cls())

    @pytest.mark.parametrize(
        ("cls", "truth", "pred", "res"),
        [
            (
                MLAccuracy,
                np.array([[0, 1], [1, 1]]),
                np.array([[0, 1], [1, 1]]),
                1,
            ),
            (
                MLAccuracy,
                np.array([[0, 1], [1, 1]]),
                np.array([[1, 0], [0, 0]]),
                0,
            ),
            (
                MLF1Macro,
                np.array([[0, 1], [1, 1]]),
                np.array([[0, 1], [1, 1]]),
                1,
            ),
            (
                MLF1Macro,
                np.array([[0, 1], [1, 1]]),
                np.array([[1, 0], [0, 0]]),
                0,
            ),
            (
                MLF1Macro,
                np.array([[0, 1, 1], [1, 1, 1], [0, 0, 1]]),
                np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]),
                0.55,
            ),
            (
                MLF1Micro,
                np.array([[0, 1], [1, 1]]),
                np.array([[0, 1], [1, 1]]),
                1,
            ),
            (
                MLF1Micro,
                np.array([[0, 1], [1, 1]]),
                np.array([[1, 0], [0, 0]]),
                0,
            ),
            (
                MLF1Micro,
                np.array([[0, 1, 1], [1, 1, 1], [0, 0, 1]]),
                np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]),
                0.73,
            ),
            (
                MLF1Weighted,
                np.array([[0, 1], [1, 1]]),
                np.array([[0, 1], [1, 1]]),
                1,
            ),
            (
                MLF1Weighted,
                np.array([[0, 1], [1, 1]]),
                np.array([[1, 0], [0, 0]]),
                0,
            ),
            (
                MLF1Micro,
                np.array([[0, 1, 1], [1, 1, 1], [0, 0, 1]]),
                np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]),
                0.72,
            ),
        ],
    )
    def test_mlc_metrics(
        self,
        cls: Type[AbstractMetric],
        truth: np.ndarray,
        pred: np.ndarray,
        res: float,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._test_metric(cls())
        with caplog.at_level(logging.WARNING):
            self._test_metric_invalid(cls())
        assert "Error computing" in caplog.text, "Warning should be logged."
        self._test_starting_metric(cls())
        self._test_comparisons(cls())
        m = cls()
        self._test_result(m, truth, pred, res)
        for idx in range(truth.shape[1]):
            np.testing.assert_almost_equal(
                m.unitary(truth[:, idx], pred[:, idx]),
                f1_score(truth[:, idx], pred[:, idx]),
            )

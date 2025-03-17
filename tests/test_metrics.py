import logging
from typing import List, Type

import numpy as np
import numpy.testing
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
    EventbasedF1,
    MLAccuracy,
    MLF1Macro,
    MLF1Micro,
    MLF1Weighted,
    SegmentbasedErrorRate,
    SegmentbasedF1,
)
from autrainer.training.utils import disaggregated_evaluation


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
    def test_classification_regression_metrics(
        self, cls: Type[AbstractMetric]
    ) -> None:
        self._test_metric(cls())
        self._test_metric_invalid(cls())
        self._test_starting_metric(cls())
        self._test_comparisons(cls())

    @pytest.mark.parametrize(
        "cls,truth,pred,res",
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
            (
                EventbasedF1,
                # 2 samples, 4 segments, 3 classes
                np.array(
                    [
                        # Sample 1: Class 0 active in segments 0-1, Class 1 active in segment 2, Class 2 inactive
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        # Sample 2: Class 0 inactive, Class 1 active in segments 1-2, Class 2 active in segment 3
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    ]
                ),
                # Predictions match ground truth exactly
                np.array(
                    [
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    ]
                ),
                1.0,  # Perfect match
            ),
            (
                EventbasedF1,
                # 2 samples, 4 segments, 3 classes
                np.array(
                    [
                        # Sample 1: Class 0 active in segments 0-1, Class 1 active in segment 2, Class 2 inactive
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        # Sample 2: Class 0 inactive, Class 1 active in segments 1-2, Class 2 active in segment 3
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    ]
                ),
                # Predictions with some errors
                np.array(
                    [
                        # Sample 1: Missed Class 0 in segment 0, detected Class 2 in segment 3 (false positive)
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                        # Sample 2: Detected Class 0 in segment 0 (false positive), missed Class 2 in segment 3
                        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    ]
                ),
                0.5,  # 3 correct events, 2 false positives, 2 false negatives
            ),
            (
                SegmentbasedF1,
                # 2 samples, 4 segments, 3 classes
                np.array(
                    [
                        # Sample 1: Class 0 active in segments 0-1, Class 1 active in segment 2, Class 2 inactive
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        # Sample 2: Class 0 inactive, Class 1 active in segments 1-2, Class 2 active in segment 3
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    ]
                ),
                # Predictions match ground truth exactly
                np.array(
                    [
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    ]
                ),
                1.0,  # Perfect match
            ),
            (
                SegmentbasedF1,
                # 2 samples, 4 segments, 3 classes
                np.array(
                    [
                        # Sample 1: Class 0 active in segments 0-1, Class 1 active in segment 2, Class 2 inactive
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        # Sample 2: Class 0 inactive, Class 1 active in segments 1-2, Class 2 active in segment 3
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    ]
                ),
                # Predictions with some errors
                np.array(
                    [
                        # Sample 1: Missed Class 0 in segment 0, detected Class 2 in segment 3 (false positive)
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                        # Sample 2: Detected Class 0 in segment 0 (false positive), missed Class 2 in segment 3
                        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    ]
                ),
                0.5,  # 3 correct segments, 2 false positives, 2 false negatives
            ),
            (
                SegmentbasedF1,
                # 2 samples, 4 segments, 3 classes
                np.array(
                    [
                        # Sample 1
                        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
                        # Sample 2
                        [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
                    ]
                ),
                # Predictions match ground truth exactly
                np.array(
                    [
                        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
                    ]
                ),
                1.0,  # Perfect match
            ),
            (
                SegmentbasedF1,
                # 2 samples, 4 segments, 3 classes
                np.array(
                    [
                        # Sample 1
                        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
                        # Sample 2
                        [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
                    ]
                ),
                # Predictions with some errors
                np.array(
                    [
                        # Sample 1
                        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        # Sample 2
                        [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]],
                    ]
                ),
                0.5,  # 3 correct segments, 2 false positives, 2 false negatives
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

    @pytest.mark.parametrize(
        "targets,predictions,indices,metrics,groundtruth,target_column,stratify,results",
        [
            (
                np.array([0, 1, 2, 3, 4]),
                np.array([2, 3, 4, 5, 6]),
                np.array([0, 1, 2, 3, 4]),
                [MAE()],
                pd.DataFrame([0, 1, 2, 3, 4], columns=["truth"]),
                "truth",
                [],
                {
                    "mae": {
                        "all": 2.0,
                    }
                },
            ),
            (
                np.array([0, 1, 2, 3, 4]),
                np.array([2, 3, 4, 6, 7]),
                np.array([0, 1, 2, 3, 4]),
                [MAE()],
                pd.DataFrame(
                    [[0, 0], [1, 0], [2, 0], [3, 1], [4, 1]],
                    columns=["truth", "foo"],
                ),
                "truth",
                ["foo"],
                {"mae": {"all": 2.4, 0: 2, 1: 3}},
            ),
            (
                np.array([0, 1, 2, 3, 4]),
                np.array([2, 3, 4, 7, 7]),
                np.array([0, 3, 4, 1, 2]),
                [MAE()],
                pd.DataFrame(
                    [[0, 0], [1, 0], [2, 0], [3, 1], [4, 1]],
                    columns=["truth", "foo"],
                ),
                "truth",
                ["foo"],
                {"mae": {"all": 2.6, 0: 3, 1: 2}},
            ),
        ],
    )
    def test_disaggregated_evaluation(
        self,
        targets,
        predictions,
        indices,
        metrics,
        groundtruth,
        target_column,
        stratify,
        results,
    ):
        res = disaggregated_evaluation(
            targets=targets,
            predictions=predictions,
            indices=indices,
            metrics=metrics,
            groundtruth=groundtruth,
            target_column=target_column,
            stratify=stratify,
        )
        assert res == results

    @pytest.mark.parametrize(
        "cls,params,truth,pred,res",
        [
            (
                EventbasedF1,
                {
                    "t_collar": 0.1,
                    "percentage_of_length": 0.1,
                    "n_segments": 10,
                    "n_classes": 2,
                },
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,  # class 0 events in segments 2-4, 7-8
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]  # class 1 events in segments 0-1, 6-7
                    ]
                ),
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                0.0,
            ),
            (
                EventbasedF1,
                {
                    "t_collar": 3.0,
                    "percentage_of_length": 0.8,
                    "n_segments": 10,
                    "n_classes": 2,
                },
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                1.0,
            ),
            (
                SegmentbasedF1,
                {"segment_length": 1.0, "n_segments": 10, "n_classes": 2},
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                0.67,
            ),
            (
                SegmentbasedF1,
                {"segment_length": 1.0, "n_segments": 10, "n_classes": 2},
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            1,
                            0,
                            1,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                0.5,
            ),
            (
                SegmentbasedErrorRate,
                {"segment_length": 1.0, "n_segments": 10, "n_classes": 2},
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                0.6,
            ),
            (
                SegmentbasedErrorRate,
                {"segment_length": 1.0, "n_segments": 10, "n_classes": 2},
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                0.0,
            ),
            (
                SegmentbasedErrorRate,
                {"segment_length": 1.0, "n_segments": 10, "n_classes": 2},
                np.array(
                    [
                        [
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                        ]
                    ]
                ),
                0.86,
            ),
        ],
    )
    def test_sed_metrics(
        self,
        cls: Type[AbstractMetric],
        params: dict,
        truth: np.ndarray,
        pred: np.ndarray,
        res: float,
    ) -> None:
        metric = cls(**params)
        result = metric(truth, pred)
        assert abs(result - res) < 0.01, f"Expected {res}, got {result}"

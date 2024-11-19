import numpy as np
import sklearn.metrics

from .abstract_metric import BaseAscendingMetric


class MLAccuracy(BaseAscendingMetric):
    def __init__(self):
        """Accuracy metric using `sklearn.metrics.accuracy_score`."""
        super().__init__(name="ml-accuracy", fn=sklearn.metrics.accuracy_score)


class MLF1Macro(BaseAscendingMetric):
    def __init__(self):
        """F1 macro metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-f1-macro",
            fn=sklearn.metrics.f1_score,
            average="macro",
        )

    def unitary(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unitary evaluation of metric.

        Metric computed with `average='binary'`
        i.e. only accounting for the positive label.

        Args:
            y_true: ground truth values.
            y_pred: prediction values.

        Returns:
            The unitary score.
        """
        return float(
            sklearn.metrics.f1_score(
                y_true=y_true, y_pred=y_pred, average="binary"
            )
        )


class MLF1Micro(BaseAscendingMetric):
    def __init__(self):
        """F1 micro metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-f1-micro",
            fn=sklearn.metrics.f1_score,
            average="micro",
        )

    def unitary(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unitary evaluation of metric.

        Metric computed with `average='binary'`
        i.e. only accounting for the positive label.

        Args:
            y_true: ground truth values.
            y_pred: prediction values.

        Returns:
            The unitary score.
        """
        return float(
            sklearn.metrics.f1_score(
                y_true=y_true, y_pred=y_pred, average="binary"
            )
        )


class MLF1Weighted(BaseAscendingMetric):
    def __init__(self):
        """F1 weighted metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-f1-weighted",
            fn=sklearn.metrics.f1_score,
            average="weighted",
        )

    def unitary(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unitary evaluation of metric.

        Metric computed with `average='binary'`
        i.e. only accounting for the positive label.

        Args:
            y_true: ground truth values.
            y_pred: prediction values.

        Returns:
            The unitary score.
        """
        return float(
            sklearn.metrics.f1_score(
                y_true=y_true, y_pred=y_pred, average="binary"
            )
        )

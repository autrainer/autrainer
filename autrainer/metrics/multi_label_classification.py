import sklearn.metrics

from .abstract_metric import BaseAscendingMetric


def _safe_f1_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
):
    multilabel = True
    if (
        (len(y_true.shape) == 1)
        or (y_true.shape[0] == 1)
        or (y_true.shape[1] == 1)
    ):
        multilabel = False
    if average != "binary" and not multilabel:
        average = "binary"
    return sklearn.metrics.f1_score(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


class MLAccuracy(BaseAscendingMetric):
    def __init__(self):
        """Accuracy metric using `sklearn.metrics.accuracy_score`."""
        super().__init__(name="ml-accuracy", fn=sklearn.metrics.accuracy_score)


class MLF1Macro(BaseAscendingMetric):
    def __init__(self):
        """F1 macro metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-f1-macro",
            fn=_safe_f1_score,
            average="macro",
        )


class MLF1Micro(BaseAscendingMetric):
    def __init__(self):
        """F1 micro metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-f1-micro",
            fn=_safe_f1_score,
            average="micro",
        )


class MLF1Weighted(BaseAscendingMetric):
    def __init__(self):
        """F1 weighted metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-f1-weighted",
            fn=_safe_f1_score,
            average="weighted",
        )

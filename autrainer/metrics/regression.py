from typing import Callable

import audmetric
import numpy as np

from .abstract_metric import BaseAscendingMetric, BaseDescendingMetric


def _mean_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    **kwargs,
) -> float:
    """Calculate the mean of a metric for each target if multi-target arrays
    are given. If single-target arrays are given, the metric is calculated
    directly.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        metric_fn: Metric function.

    Returns:
        Mean of the metric for each target.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.ndim == 1:
        return metric_fn(y_true, y_pred, **kwargs)
    m = np.empty(y_true.shape[1])
    for i in range(y_true.shape[1]):
        m[i] = metric_fn(y_true[:, i], y_pred[:, i], **kwargs)
    return np.mean(m)


class MSE(BaseDescendingMetric):
    def __init__(self):
        """Mean squared error metric using `audmetric.mean_squared_error`
        for (multi-target) regression. The metric is calculated for each
        target separately and the mean is returned.
        """
        super().__init__(
            name="mse",
            fn=_mean_metric,
            metric_fn=audmetric.mean_squared_error,
        )


class MAE(BaseDescendingMetric):
    def __init__(self):
        """Mean absolute error metric using `audmetric.mean_absolute_error`
        for (multi-target) regression. The metric is calculated for each
        target separately and the mean is returned.
        """
        super().__init__(
            name="mae",
            fn=_mean_metric,
            metric_fn=audmetric.mean_absolute_error,
        )


class PCC(BaseAscendingMetric):
    def __init__(self):
        """Pearson correlation coefficient metric using
        `audmetric.pearson_cc` for (multi-target) regression. The metric is
        calculated for each target separately and the mean is returned.
        """
        super().__init__(
            name="pcc",
            fn=_mean_metric,
            metric_fn=audmetric.pearson_cc,
        )


class CCC(BaseAscendingMetric):
    def __init__(self):
        """Concordance correlation coefficient metric using
        `audmetric.concordance_cc` for (multi-target) regression. The metric is
        calculated for each target separately and the mean is returned.
        """
        super().__init__(
            name="ccc",
            fn=_mean_metric,
            metric_fn=audmetric.concordance_cc,
        )

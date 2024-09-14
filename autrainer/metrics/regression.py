import audmetric

from .abstract_metric import BaseAscendingMetric, BaseDescendingMetric


class MSE(BaseDescendingMetric):
    def __init__(self):
        """Mean squared error metric using `audmetric.mean_squared_error`."""
        super().__init__(name="mse", fn=audmetric.mean_squared_error)


class MAE(BaseDescendingMetric):
    def __init__(self):
        """Mean absolute error metric using `audmetric.mean_absolute_error`."""
        super().__init__(name="mae", fn=audmetric.mean_absolute_error)


class PCC(BaseAscendingMetric):
    def __init__(self):
        """Pearson correlation coefficient metric using
        `audmetric.pearson_cc`.
        """
        super().__init__(name="pcc", fn=audmetric.pearson_cc)


class CCC(BaseAscendingMetric):
    def __init__(self):
        """Concordance correlation coefficient metric using
        `audmetric.concordance_cc`.
        """
        super().__init__(name="ccc", fn=audmetric.concordance_cc)

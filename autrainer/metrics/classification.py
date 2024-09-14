import audmetric

from .abstract_metric import BaseAscendingMetric


class Accuracy(BaseAscendingMetric):
    def __init__(self):
        """Accuracy metric using `audmetric.accuracy`."""
        super().__init__(name="accuracy", fn=audmetric.accuracy)


class UAR(BaseAscendingMetric):
    def __init__(self):
        """Unweighted average recall metric using
        `audmetric.unweighted_average_recall`.
        """
        super().__init__(name="uar", fn=audmetric.unweighted_average_recall)


class F1(BaseAscendingMetric):
    def __init__(self):
        """F1 metric using `audmetric.unweighted_average_fscore`."""
        super().__init__(name="f1", fn=audmetric.unweighted_average_fscore)

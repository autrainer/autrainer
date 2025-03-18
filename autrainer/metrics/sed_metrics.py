import numpy as np
from sed_eval.sound_event import EventBasedMetrics

from .abstract_metric import BaseAscendingMetric


class SegmentBasedF1(BaseAscendingMetric):
    def __init__(self, target_transform):
        """F1 macro metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="segment-based-f1",
            fn=self.forward,
        )
        self.target_transform = target_transform

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        sed = EventBasedMetrics(self.target_transform.labels)
        for i, x in enumerate(y_pred):
            res = self.target_transform.decode(x)
            preds = []
            for element in res:
                element["file"] = i
                preds.append(element)
            res = self.target_transform.decode(y_true[i])
            truth = []
            for element in res:
                element["file"] = i
                truth.append(element)
            sed.evaluate(
                reference_event_list=truth, estimated_event_list=preds
            )
        return float(sed.results_overall_metrics()["f_measure"]["f_measure"])

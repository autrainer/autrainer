import numpy as np
from sed_eval.sound_event import EventBasedMetrics, SegmentBasedMetrics

from autrainer.datasets.utils.target_transforms import SEDEncoder

from .abstract_metric import BaseAscendingMetric, BaseDescendingMetric


class SEDMetric:
    def __init__(
        self,
        target_transform: SEDEncoder,
        t_collar: float,
        percentage_of_length: float,
        time_resolution: float,
        metric_type: str = "event",
    ):
        """SED Metric Mixin Class for Event-based and Segment-based metrics.
        Uses `sed_eval.sound_event.{EventBasedMetrics, SegmentBasedMetrics}`.

        TODO: avoid recomputation of metrics for each metric instance.

        Args:
            target_transform: The SED encoder to use for decoding.
            metric_type: Type of metric to use ('event' or 'segment').
            **kwargs: Additional parameters for specific metric types:
                     - event: t_collar, percentage_of_length
                     - segment: time_resolution
        """
        if not target_transform.labels:
            raise ValueError("target_transform must have at least one label")

        if metric_type not in ["event", "segment"]:
            raise ValueError(
                f"metric_type must be 'event' or 'segment', got {metric_type}"
            )

        self.target_transform = target_transform
        self.t_collar = t_collar
        self.percentage_of_length = percentage_of_length
        self.time_resolution = time_resolution
        self.metric_type = metric_type
        self.implemented_metrics = ["f_measure", "error_rate"]
        self._init_sed_metric()

    def _init_sed_metric(self) -> None:
        """Initialize sed_eval metric based on type."""
        if self.metric_type == "event":
            self._sed = EventBasedMetrics(
                event_label_list=self.target_transform.labels,
                t_collar=self.t_collar,
                percentage_of_length=self.percentage_of_length,
            )
        else:
            self._sed = SegmentBasedMetrics(
                event_label_list=self.target_transform.labels,
                time_resolution=self.time_resolution,
            )

    def _evaluate_events(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Evaluate matching events of the target and prediction arrays."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if not (y_true.ndim in [2, 3] and y_pred.ndim in [2, 3]):
            raise ValueError(
                f"Inputs must have 2 or 3 dimensions, got y_true: {y_true.ndim}, y_pred: {y_pred.ndim}"
            )

        self._sed.reset()

        y_true = y_true[np.newaxis, ...] if y_true.ndim == 2 else y_true
        y_pred = y_pred[np.newaxis, ...] if y_pred.ndim == 2 else y_pred

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            true_events = self.target_transform.decode(true)
            pred_events = self.target_transform.decode(pred)
            for event in true_events + pred_events:
                event["file"] = i
            self._sed.evaluate(  # unified sed evaluator
                reference_event_list=true_events,
                estimated_event_list=pred_events,
            )

    def get_metric(self, metric_type: str) -> float:
        """Get the metric value."""
        if metric_type not in self.implemented_metrics:
            raise ValueError(
                f"metric_type must be one of {self.implemented_metrics}, got {metric_type}"
            )

        results = self._sed.results_overall_metrics()
        if metric_type == "error_rate" and self.metric_type == "segment":
            return 1.0 - float(results["f_measure"]["f_measure"])

        return float(results[metric_type][metric_type])


class SegmentBasedF1(BaseAscendingMetric, SEDMetric):
    def __init__(self, target_transform: SEDEncoder, **kwargs):
        """Segment-based F1 metric using `sed_eval.sound_event.SegmentBasedMetrics`."""
        SEDMetric.__init__(
            self,
            target_transform=target_transform,
            metric_type="segment",
        )
        BaseAscendingMetric.__init__(
            self,
            name="segment-based-f1",
            fn=self.forward,
            fallback=-1e32,
            **kwargs,
        )
        self.metric_name = "f_measure"

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self._evaluate_events(y_true, y_pred)
        return self.get_metric(self.metric_name)


class EventBasedF1(BaseAscendingMetric, SEDMetric):
    def __init__(self, target_transform: SEDEncoder, **kwargs):
        """Event-based F1 metric using `sed_eval.sound_event.EventBasedMetrics`."""
        SEDMetric.__init__(
            self,
            target_transform=target_transform,
            metric_type="event",
        )
        BaseAscendingMetric.__init__(
            self,
            name="event-based-f1",
            fn=self.forward,
            fallback=-1e32,
            **kwargs,
        )
        self.metric_name = "f_measure"

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self._evaluate_events(y_true, y_pred)
        return self.get_metric(self.metric_name)


class SegmentBasedErrorRate(BaseDescendingMetric, SEDMetric):
    def __init__(self, target_transform: SEDEncoder, **kwargs):
        """Segment-based error rate metric using `sed_eval.sound_event.SegmentBasedMetrics`."""
        SEDMetric.__init__(
            self,
            target_transform=target_transform,
            metric_type="segment",
        )
        BaseDescendingMetric.__init__(
            self,
            name="segment-based-error-rate",
            fn=self.forward,
            fallback=1e32,
            **kwargs,
        )
        self.metric_name = "error_rate"

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self._evaluate_events(y_true, y_pred)
        return self.get_metric(self.metric_name)

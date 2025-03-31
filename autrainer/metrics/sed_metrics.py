import numpy as np
from sed_eval.sound_event import EventBasedMetrics, SegmentBasedMetrics

from ..datasets.utils.target_transforms.sed_encoder import SEDEncoder
from .abstract_metric import BaseAscendingMetric, BaseDescendingMetric


class BaseSEDMetricMixin:
    def __init__(
        self,
        target_transform: SEDEncoder,
        metric_type: str = "event",
        t_collar: float = 0.200,
        percentage_of_length: float = 0.2,
        time_resolution: float = 1.0,
    ):
        """Initialize SED metric parameters.

        Args:
            target_transform: The SED encoder to use for decoding.
            metric_type: Type of metric to use.
            t_collar: Time collar for matching events in seconds.
            percentage_of_length: Percentage of length tolerance for matching events.
            time_resolution: Time resolution for segment-based metrics.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If target_transform is invalid or parameters are out of range.
        """
        if not target_transform.labels:
            raise ValueError("target_transform must have at least one label")
        if t_collar <= 0:
            raise ValueError(f"t_collar must be positive, got {t_collar}")
        if percentage_of_length <= 0:
            raise ValueError(
                f"percentage_of_length must be positive, got {percentage_of_length}"
            )

        self.target_transform = target_transform
        self.metric_type = metric_type
        self.t_collar = t_collar
        self.percentage_of_length = percentage_of_length
        self.time_resolution = time_resolution
        self.implemented_metrics = ["f_measure", "error_rate"]
        if self.metric_type == "event":
            self._sed = EventBasedMetrics(
                event_label_list=self.target_transform.labels,
                t_collar=self.t_collar,
                percentage_of_length=self.percentage_of_length,
            )
        else:  # segment
            self._sed = SegmentBasedMetrics(
                event_label_list=self.target_transform.labels,
                time_resolution=self.time_resolution,
            )

    def _evaluate_events(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
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
            self._sed.evaluate(
                reference_event_list=true_events,
                estimated_event_list=pred_events,
            )

    def get_metric(self, metric_type: str) -> float:
        if metric_type not in self.implemented_metrics:
            raise ValueError(
                f"metric_type must be one of {self.implemented_metrics}, got {metric_type}"
            )

        results = self._sed.results_overall_metrics()
        if metric_type == "error_rate" and self.metric_type == "segment":
            return 1.0 - float(results["f_measure"]["f_measure"])

        return float(results[metric_type][metric_type])


class SegmentBasedF1(BaseAscendingMetric, BaseSEDMetricMixin):
    def __init__(self, target_transform: SEDEncoder, **kwargs):
        """Segment-based F1 metric using `sed_eval.sound_event.SegmentBasedMetrics`."""
        BaseSEDMetricMixin.__init__(
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


class EventBasedF1(BaseAscendingMetric, BaseSEDMetricMixin):
    def __init__(self, target_transform: SEDEncoder, **kwargs):
        """Event-based F1 metric using `sed_eval.sound_event.SegmentBasedMetrics`."""
        BaseSEDMetricMixin.__init__(
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


class SegmentBasedErrorRate(BaseDescendingMetric, BaseSEDMetricMixin):
    def __init__(self, target_transform: SEDEncoder, **kwargs):
        """Segment-based error rate metric using `sed_eval.sound_event.SegmentBasedMetrics`."""
        BaseSEDMetricMixin.__init__(
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

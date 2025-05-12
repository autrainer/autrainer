from collections import OrderedDict
import hashlib

import numpy as np


try:
    from sed_eval.sound_event import EventBasedMetrics, SegmentBasedMetrics

    SED_EVAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    SED_EVAL_AVAILABLE = False

from autrainer.datasets.utils.target_transforms import SEDEncoder

from .abstract_metric import BaseAscendingMetric, BaseDescendingMetric


class SEDMetricBackend:
    _instance = None

    def __new__(cls):
        """Create a new instance of the SEDMetricBackend."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self) -> None:
        """Initialize the backend with default values."""
        if not SED_EVAL_AVAILABLE:
            raise ImportError(
                "sed_eval is required for SED metrics. Install it with: poetry install --with sed_eval"
            )
        self.cache = OrderedDict()
        self.max_cache_size = 1
        self._sed = None
        self.target_transform = None
        self.metric_type = None
        self.t_collar = None
        self.percentage_of_length = None
        self.time_resolution = None

    def configure(
        self,
        target_transform: SEDEncoder,
        t_collar: float = 0.2,
        percentage_of_length: float = 0.5,
        time_resolution: float = 0.01,
        metric_type: str = "event",
    ) -> None:
        """Configure the SED metric backend.

        Args:
            target_transform: SED encoder for target transformation.
            t_collar: Time collar for event-based metrics in seconds.
            percentage_of_length: Percentage of event length for matching.
            time_resolution: Time resolution for segment-based metrics in seconds.
            metric_type: Type of metric to use ('event' or 'segment').

        Raises:
            ValueError: If parameters are invalid or target_transform has no labels.
            ImportError: If sed_eval is not installed.
        """
        if not SED_EVAL_AVAILABLE:
            raise ImportError(
                "sed_eval is required for SED metrics. Install it with: poetry install --with sed_eval"
            )

        if not target_transform.labels:
            raise ValueError("target_transform must have at least one label")

        if t_collar <= 0:
            raise ValueError("t_collar must be positive")

        if not 0 < percentage_of_length <= 1:
            raise ValueError("percentage_of_length must be between 0 and 1")

        if time_resolution <= 0:
            raise ValueError("time_resolution must be positive")

        if metric_type not in ["event", "segment"]:
            raise ValueError("metric_type must be either 'event' or 'segment'")

        self.cache.clear()
        self.target_transform = target_transform
        self.t_collar = t_collar
        self.percentage_of_length = percentage_of_length
        self.time_resolution = time_resolution
        self.metric_type = metric_type
        self._init_sed_metric()

    def _init_sed_metric(self) -> None:
        """Initialize the SED metric based on the configured type."""
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
        """Evaluate events using the configured SED metric."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if not (y_true.ndim in [2, 3] and y_pred.ndim in [2, 3]):
            raise ValueError(
                f"Inputs must have 2 or 3 dimensions, got y_true: {y_true.ndim}, y_pred: {y_pred.ndim}"
            )

        self._sed.reset()

        y_true = y_true[np.newaxis, ...] if y_true.ndim == 2 else y_true
        y_pred = y_pred[np.newaxis, ...] if y_pred.ndim == 2 else y_pred

        try:
            for i, (true, pred) in enumerate(zip(y_true, y_pred)):
                true_events = self.target_transform.decode(true)
                pred_events = self.target_transform.decode(pred)
                for event in true_events + pred_events:
                    event["file"] = i
                self._sed.evaluate(
                    reference_event_list=true_events,
                    estimated_event_list=pred_events,
                )
        except Exception as e:
            raise ValueError(f"Failed to evaluate events: {str(e)}")

    def calculate(self, targets: np.ndarray, preds: np.ndarray) -> dict:
        """Calculate metrics for the given targets and predictions."""
        self._evaluate_events(targets, preds)
        results = self._sed.results_overall_metrics()
        f_measure = float(results["f_measure"]["f_measure"])
        error_rate = (
            1.0 - f_measure
            if self.metric_type == "segment"
            else float(results["error_rate"]["error_rate"])
        )
        return {"f_measure": f_measure, "error_rate": error_rate}

    def __call__(
        self, targets: np.ndarray, preds: np.ndarray, metric: str
    ) -> float:
        """Calculate a specific metric for the given targets and predictions."""
        if metric not in ["f_measure", "error_rate"]:
            raise ValueError(
                "metric must be either 'f_measure' or 'error_rate'"
            )

        key = hashlib.md5(
            np.concatenate([targets.flatten(), preds.flatten()]).tobytes()
        ).hexdigest()

        if key not in self.cache:
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)
            self.cache[key] = self.calculate(targets, preds)
        return self.cache[key][metric]


class SegmentBasedF1(BaseAscendingMetric):
    def __init__(self, target_transform: SEDEncoder, **kwargs):
        backend = SEDMetricBackend()
        backend.configure(
            target_transform=target_transform, metric_type="segment", **kwargs
        )
        super().__init__(
            name="segment-based-f1",
            fn=backend,
            metric="f_measure",
            fallback=-1e32,
        )


class EventBasedF1(BaseAscendingMetric):
    def __init__(self, target_transform: SEDEncoder, **kwargs):
        backend = SEDMetricBackend()
        backend.configure(
            target_transform=target_transform, metric_type="event", **kwargs
        )
        super().__init__(
            name="event-based-f1",
            fn=backend,
            metric="f_measure",
            fallback=-1e32,
        )


class SegmentBasedErrorRate(BaseDescendingMetric):
    def __init__(self, target_transform: SEDEncoder, **kwargs):
        backend = SEDMetricBackend()
        backend.configure(
            target_transform=target_transform, metric_type="segment", **kwargs
        )
        super().__init__(
            name="segment-based-error-rate",
            fn=backend,
            metric="error_rate",
            fallback=1e32,
        )

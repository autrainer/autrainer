import hashlib
import threading
from typing import Dict, List

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

        # Initialize cache-related attributes
        self._cache_lock = threading.Lock()
        self._decode_cache: Dict[str, List[Dict]] = {}
        self._eval_cache: Dict[str, Dict] = {}
        self._maxsize = 128  # Maximum cache size

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

    def _decode_with_cache(
        self, arr: np.ndarray, file_idx: int = 0
    ) -> List[Dict]:
        """Decode array to events with caching.

        Args:
            arr: Input array to decode
            file_idx: File index to assign to events

        Returns:
            List of decoded events
        """
        arr_hash = hashlib.sha1(arr.tobytes()).hexdigest()

        with self._cache_lock:
            events = self._decode_cache.get(arr_hash)
            if events is None:
                events = self.target_transform.decode(arr)
                if len(self._decode_cache) >= self._maxsize:
                    self._decode_cache.pop(next(iter(self._decode_cache)))
                self._decode_cache[arr_hash] = events.copy()
            else:
                events = events.copy()

        # Add file index to events
        for event in events:
            event["file"] = file_idx

        return events

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Compute all SED metrics for the given arrays using caching.

        This method handles the core computation of SED metrics including:
        - Converting arrays to event lists
        - Caching intermediate results
        - Running the actual evaluation using sed_eval

        Args:
            y_true: Ground truth array
            y_pred: Predicted array
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if not (y_true.ndim in [2, 3] and y_pred.ndim in [2, 3]):
            raise ValueError(
                f"Inputs must have 2 or 3 dimensions, got y_true: {y_true.ndim}, y_pred: {y_pred.ndim}"
            )

        # Create cache key from both array hashes
        true_hash = hashlib.sha1(y_true.tobytes()).hexdigest()
        pred_hash = hashlib.sha1(y_pred.tobytes()).hexdigest()
        cache_key = f"{true_hash}_{pred_hash}_{self.metric_type}"

        # Check if evaluation result is cached
        with self._cache_lock:
            cached_result = self._eval_cache.get(cache_key)

        if cached_result is not None:
            # Reset metrics and evaluate with cached results
            self._sed.reset()
            for i, (true, pred) in enumerate(zip(y_true, y_pred)):
                true_events = self._decode_with_cache(true, i)
                pred_events = self._decode_with_cache(pred, i)
                self._sed.evaluate(
                    reference_event_list=true_events,
                    estimated_event_list=pred_events,
                )
            return

        self._sed.reset()
        y_true = y_true[np.newaxis, ...] if y_true.ndim == 2 else y_true
        y_pred = y_pred[np.newaxis, ...] if y_pred.ndim == 2 else y_pred

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            true_events = self._decode_with_cache(true, i)
            pred_events = self._decode_with_cache(pred, i)

            self._sed.evaluate(
                reference_event_list=true_events,
                estimated_event_list=pred_events,
            )

        # Cache the evaluation result
        with self._cache_lock:
            if len(self._eval_cache) >= self._maxsize:
                self._eval_cache.pop(next(iter(self._eval_cache)))
            self._eval_cache[cache_key] = (
                self._sed.results_overall_metrics().copy()
            )

    def get_metric_value(self, metric_type: str) -> float:
        """Get a specific metric value from the computed results.

        Args:
            metric_type: Type of metric to get (e.g. 'f_measure' or 'error_rate')

        Returns:
            The value of the requested metric

        Raises:
            ValueError: If metric_type is not one of the implemented metrics
        """
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
        self._compute_metrics(y_true, y_pred)
        return self.get_metric_value(self.metric_name)


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
        self._compute_metrics(y_true, y_pred)
        return self.get_metric_value(self.metric_name)


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
        self._compute_metrics(y_true, y_pred)
        return self.get_metric_value(self.metric_name)

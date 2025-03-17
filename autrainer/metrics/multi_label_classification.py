import numpy as np
import sklearn.metrics

from .abstract_metric import BaseAscendingMetric, BaseDescendingMetric


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
            zero_division=0,
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
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=0
            )
        )


class MLF1Micro(BaseAscendingMetric):
    def __init__(self):
        """F1 micro metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-f1-micro",
            fn=sklearn.metrics.f1_score,
            average="micro",
            zero_division=0,
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
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=0
            )
        )


class MLF1Weighted(BaseAscendingMetric):
    def __init__(self):
        """F1 weighted metric using `sklearn.metrics.f1_score`."""
        super().__init__(
            name="ml-f1-weighted",
            fn=sklearn.metrics.f1_score,
            average="weighted",
            zero_division=0,
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
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=0
            )
        )


class EventbasedF1(BaseAscendingMetric):
    def __init__(
        self,
        t_collar=0.200,
        percentage_of_length=0.2,
        segment_duration=0.25,
        n_segments=40,
        n_classes=10,
    ):
        """Event-based F1 score for sound event detection. Implemented as in:
        https://github.com/turpaultn/DCASE2019_task4/blob/public/baseline/evaluation_measures.py

        Args:
            t_collar: Time collar for matching events in seconds.
            percentage_of_length: Percentage of length tolerance for matching events.
            median_filter_width: Width of the median filter to apply to the predictions.
            segment_duration: Duration of each segment in seconds.
            n_segments: Number of segments in the input data.
            n_classes: Number of classes in the input data.
        """
        super().__init__(
            name="event-based-f1",
            fn=self.compute_f1,
        )
        self.t_collar = t_collar
        self.percentage_of_length = percentage_of_length
        self.segment_duration = segment_duration
        self.n_segments = n_segments
        self.n_classes = n_classes

    def compute_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute event-based F1 score."""
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(1, -1)
            y_pred = y_pred.reshape(1, -1)

        if len(y_true.shape) == 2:
            segments_per_class = y_true.shape[1] // self.n_classes
            y_true = np.asarray(
                [
                    y_true[
                        :,
                        c * segments_per_class : (c + 1) * segments_per_class,
                    ]
                    for c in range(self.n_classes)
                ]
            ).transpose(1, 2, 0)
            y_pred = np.asarray(
                [
                    y_pred[
                        :,
                        c * segments_per_class : (c + 1) * segments_per_class,
                    ]
                    for c in range(self.n_classes)
                ]
            ).transpose(1, 2, 0)

        total_tp = total_fp = total_fn = 0
        for class_idx in range(y_true.shape[2]):
            true_events = self._extract_events(y_true)[class_idx]
            pred_events = self._extract_events(y_pred)[class_idx]
            if not true_events and not pred_events:
                continue
            else:
                tp = len(
                    self._match_events_with_collar(true_events, pred_events)
                )
                total_tp += tp
                total_fp += len(pred_events) - tp
                total_fn += len(true_events) - tp

        denominator = total_tp + 0.5 * (total_fp + total_fn)
        return total_tp / denominator if denominator > 0 else 0.0

    def _extract_events(self, y: np.ndarray) -> dict:
        if hasattr(y, "cpu"):
            y = y.cpu().numpy()

        events_by_class = {i: [] for i in range(y.shape[2])}
        for b in range(y.shape[0]):
            for c in range(y.shape[2]):
                activity = y[b, :, c]
                padded = np.concatenate([[0], activity, [0]])
                transitions = np.diff(padded)
                starts = np.where(transitions == 1)[0]
                ends = np.where(transitions == -1)[0] - 1
                if len(starts) > 0:
                    events_by_class[c].extend(list(zip(starts, ends)))

        return events_by_class

    def _match_events_with_collar(self, true_events, pred_events):
        matched = []
        t_events = [
            (s * self.segment_duration, e * self.segment_duration)
            for s, e in true_events
        ]
        p_events = [
            (s * self.segment_duration, e * self.segment_duration)
            for s, e in pred_events
        ]

        for i, (t_on, t_off) in enumerate(t_events):
            t_len = t_off - t_on
            threshold = max(self.t_collar, self.percentage_of_length * t_len)

            for j, (p_on, p_off) in enumerate(p_events):
                if any(m[1] == j for m in matched):
                    continue
                if (
                    abs(t_on - p_on) <= self.t_collar
                    and abs(t_off - p_off) <= threshold
                ):
                    matched.append((i, j))
                    break

        return matched


class SegmentbasedF1(BaseAscendingMetric):
    def __init__(self, segment_length=0.25, n_segments=40, n_classes=10):
        """Segment-based F1 score for sound event detection as used in DCASE 2019 Task 4.

        Args:
            segment_length: Length of segments in seconds (default: 1.0 second as per DCASE evaluation)
            feature_rate: Number of frames per second in the input data (default: 10Hz)
            median_filter_width: Width of the median filter to apply to the predictions
            n_segments: Number of segments in the input data.
            n_classes: Number of classes in the input data.
        """
        super().__init__(
            name="segment-based-f1",
            fn=self.compute_f1,
        )
        self.segment_length = segment_length
        self.n_segments = n_segments
        self.n_classes = n_classes

    def compute_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute segment-based F1 score using specified segment length."""
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(1, -1)
            y_pred = y_pred.reshape(1, -1)

        if len(y_true.shape) == 2:
            segments_per_class = y_true.shape[1] // self.n_classes
            y_true = np.asarray(
                [
                    y_true[
                        :,
                        c * segments_per_class : (c + 1) * segments_per_class,
                    ]
                    for c in range(self.n_classes)
                ]
            ).transpose(1, 2, 0)
            y_pred = np.asarray(
                [
                    y_pred[
                        :,
                        c * segments_per_class : (c + 1) * segments_per_class,
                    ]
                    for c in range(self.n_classes)
                ]
            ).transpose(1, 2, 0)

        total_tp = np.sum(np.sum((y_true == 1) & (y_pred == 1), axis=(0, 1)))
        total_fp = np.sum(np.sum((y_true == 0) & (y_pred == 1), axis=(0, 1)))
        total_fn = np.sum(np.sum((y_true == 1) & (y_pred == 0), axis=(0, 1)))
        return (
            2 * total_tp / (2 * total_tp + total_fp + total_fn)
            if 2 * total_tp + total_fp + total_fn > 0
            else 0.0
        )


class SegmentbasedErrorRate(BaseDescendingMetric):
    def __init__(self, segment_length=0.25, n_segments=40, n_classes=10):
        """Segment-based error rate for sound event detection as defined in DCASE evaluation.

        Calculates the error rate as described in [Poliner2007] over all test data based on the total
        number of insertions, deletions and substitutions.

        Args:
            segment_length: Length of segments in seconds
            n_segments: Number of segments in the input data
            n_classes: Number of classes in the input data
        """
        super().__init__(
            name="segment-based-error-rate",
            fn=self.compute_error_rate,
        )
        self.segment_length = segment_length
        self.n_segments = n_segments
        self.n_classes = n_classes

    def compute_error_rate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compute segment-based error rate according to Poliner2007.

        Error rate = (S + D + I) / N, where:
        S = Substitution errors (missed detections)
        D = Deletion errors (missed detections)
        I = Insertion errors (false alarms)
        N = Total number of ground truth segments

        Note: For DCASE evaluation, the formula is modified to:
        Error rate = (D + I) / (2*N) to ensure error rate is between 0 and 1.
        """
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(1, -1)
            y_pred = y_pred.reshape(1, -1)

        if len(y_true.shape) == 2:
            segments_per_class = y_true.shape[1] // self.n_classes
            y_true = np.asarray(
                [
                    y_true[
                        :,
                        c * segments_per_class : (c + 1) * segments_per_class,
                    ]
                    for c in range(self.n_classes)
                ]
            ).transpose(1, 2, 0)
            y_pred = np.asarray(
                [
                    y_pred[
                        :,
                        c * segments_per_class : (c + 1) * segments_per_class,
                    ]
                    for c in range(self.n_classes)
                ]
            ).transpose(1, 2, 0)

        deletions = np.sum(np.sum((y_true == 1) & (y_pred == 0), axis=(0, 1)))
        insertions = np.sum(np.sum((y_true == 0) & (y_pred == 1), axis=(0, 1)))
        total_segments = np.sum(np.sum(y_true, axis=(0, 1)))
        return (
            (deletions + insertions) / (total_segments)
            if total_segments > 0
            else 0.0
        )

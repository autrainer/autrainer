from abc import ABC, abstractmethod
import functools
import logging
from typing import Callable, Union
import warnings

import numpy as np
import pandas as pd


def ignore_runtime_warning(func):
    @functools.wraps(func)
    def wrapper_ignore_warning(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return func(*args, **kwargs)

    return wrapper_ignore_warning


class AbstractMetric(ABC):
    def __init__(
        self,
        name: str,
        fn: Callable,
        fallback: float,
        **fn_kwargs: dict,
    ) -> None:
        """Abstract class for metrics.

        Args:
            name: The name of the metric.
            fn: The function to compute the metric.
            fallback: The fallback value if the metric is NaN.
            **fn_kwargs: Additional keyword arguments to pass to the function.
        """
        self.name = name
        self.fn = fn
        self.fallback = fallback
        self.fn_kwargs = fn_kwargs
        self.logger = logging.getLogger(__name__)

    @ignore_runtime_warning
    def __call__(self, *args, **kwargs) -> float:
        """Compute the metric.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The score.
        """
        try:
            score = self.fn(*args, **kwargs, **self.fn_kwargs)
        except ValueError as e:
            self.logger.warning(
                f"Error computing {self.name} metric due to: {e}. "
                f"Defaulting to fallback value of {self.fallback}."
            )
            return self.fallback
        if np.isnan(score):
            self.logger.warning(
                f"Error computing {self.name} metric as score is NaN. "
                f"Defaulting to fallback value of {self.fallback}."
            )
            return self.fallback
        return float(score)

    @property
    @abstractmethod
    def starting_metric(self) -> float:
        """The starting metric value.

        Returns:
            The starting metric value.
        """

    @property
    @abstractmethod
    def suffix(self) -> str:
        """The suffix of the metric.

        Returns:
            The suffix of the metric.
        """

    @staticmethod
    @abstractmethod
    def get_best(a: Union[pd.Series, np.ndarray]) -> float:
        """Get the best metric value from a series of scores.

        Args:
            a: Pandas series or numpy array of scores.

        Returns:
            Best metric value.
        """

    @staticmethod
    @abstractmethod
    def get_best_pos(a: Union[pd.Series, np.ndarray]) -> int:
        """Get the position of the best metric value from a series of scores.

        Args:
            a: Pandas series or numpy array of scores.

        Returns:
            Position of the best metric value.
        """

    @staticmethod
    @abstractmethod
    def compare(a: Union[int, float], b: Union[int, float]) -> bool:
        """Compare two scores and return True if the first score is better.

        Args:
            a: First score.
            b: Second score.

        Returns:
            True if the first score is better.
        """

    def unitary(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unitary evaluation of metric.

        Metric computed for each individual label
        in the case of multilabel classification
        or regression.

        In the default case, computes the metric
        in the same way as globally.

        Args:
            y_true: ground truth values.
            y_pred: prediction values.

        Returns:
            The unitary score.
        """
        return self.__call__(y_true, y_pred)


class BaseAscendingMetric(AbstractMetric):
    def __init__(
        self,
        name: str,
        fn: Callable,
        fallback: float = None,
        **fn_kwargs: dict,
    ) -> None:
        """Base for ascending metrics with higher values being better.

        Args:
            name: The name of the metric.
            fn: The function to compute the metric.
            fallback: The fallback value if the metric is NaN. If None,
                the fallback value is set to -1e32. Defaults to None.
            **fn_kwargs: Additional keyword arguments to pass to the function.
        """
        super().__init__(name, fn, fallback or -1e32, **fn_kwargs)

    @property
    def starting_metric(self) -> float:
        """Ascending metric starting value.

        Returns:
            -1e32
        """
        return -1e32

    @property
    def suffix(self) -> str:
        """Ascending metric suffix.

        Returns:
            "max"
        """
        return "max"

    @staticmethod
    def get_best(a: Union[pd.Series, np.ndarray]) -> float:
        return float(a.max())

    @staticmethod
    def get_best_pos(a: Union[pd.Series, np.ndarray]) -> int:
        if isinstance(a, pd.Series):
            return int(a.idxmax())
        return int(a.argmax())

    @staticmethod
    def compare(a: Union[int, float], b: Union[int, float]) -> bool:
        return a > b


class BaseDescendingMetric(AbstractMetric):
    def __init__(
        self,
        name: str,
        fn: Callable,
        fallback: float = None,
        **fn_kwargs: dict,
    ) -> None:
        """Base for descending metrics with lower values being better.

        Args:
            name: The name of the metric.
            fn: The function to compute the metric.
            fallback: The fallback value if the metric is NaN. If None,
                the fallback value is set to 1e32. Defaults to None.
            **fn_kwargs: Additional keyword arguments to pass to the function.
        """
        super().__init__(name, fn, fallback or 1e32, **fn_kwargs)

    @property
    def starting_metric(self) -> float:
        """Descending metric starting value.

        Returns:
            1e32
        """
        return 1e32

    @property
    def suffix(self) -> str:
        """Descending metric suffix.

        Returns:
            "min"
        """
        return "min"

    @staticmethod
    def get_best(a: Union[pd.Series, np.ndarray]) -> float:
        return float(a.min())

    @staticmethod
    def get_best_pos(a: Union[pd.Series, np.ndarray]) -> int:
        if isinstance(a, pd.Series):
            return int(a.idxmin())
        return int(a.argmin())

    @staticmethod
    def compare(a: Union[int, float], b: Union[int, float]) -> bool:
        return a < b

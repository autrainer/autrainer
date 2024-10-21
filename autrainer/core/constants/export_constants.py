from typing import Dict, List, Union

from .abstract_constants import AbstractConstants


class ExportConstants(AbstractConstants):
    """Singleton for managing the export and logging configurations of
    `autrainer`."""

    _name = "ExportConstants"
    _logging_depth = 2
    _ignore_params = [
        "results_dir",
        "experiment_id",
        "model.dataset",
        "training_type",
        "save_frequency",
        "dataset.metrics",
        "plotting",
        "model.transform",
        "dataset.transform",
        "augmentation.steps",
        "loggers",
        "progress_bar",
        "continue_training",
        "remove_continued_runs",
        "save_train_outputs",
        "save_dev_outputs",
        "save_test_outputs",
    ]
    _artifacts = [
        "model_summary.txt",
        "metrics.csv",
        {"config.yaml": ".hydra"},
    ]

    @property
    def LOGGING_DEPTH(self) -> int:
        """Get the depth of logging for configuration parameters.
        Defaults to :attr:`2`.

        Returns:
            Depth of logging for configuration parameters.
        """
        return self._logging_depth

    @LOGGING_DEPTH.setter
    def LOGGING_DEPTH(self, logging_depth: int) -> None:
        """Set the depth of logging for configuration parameters.

        Args:
            logging_depth: Depth of logging for configuration parameters.

        Raises:
            ValueError: If the logging depth is not an integer or is negative.
        """
        self._assert_type(logging_depth, int, "LOGGING_DEPTH")
        if logging_depth < 0:
            raise ValueError("LOGGING_DEPTH must be non-negative.")
        self._logging_depth = logging_depth

    @property
    def IGNORE_PARAMS(self) -> List[str]:
        """Get the ignored configuration parameters for logging.
        Defaults to :attr:`["results_dir", "experiment_id", "model.dataset",
        "training_type", "save_frequency", "dataset.metrics", "plotting",
        "model.transform", "dataset.transform", "augmentation.steps",
        "loggers", "progress_bar", "continue_training",
        "remove_continued_runs", "save_train_outputs", "save_dev_outputs",
        "save_test_outputs"]`.

        Returns:
            Ignored configuration parameters for logging.
        """
        return self._ignore_params

    @IGNORE_PARAMS.setter
    def IGNORE_PARAMS(self, ignore_params: List[str]) -> None:
        """Set the ignored configuration parameters for logging.

        Args:
            ignore_params: Ignored configuration parameters for logging.

        Raises:
            ValueError: If the ignored parameters are not a list of strings.
        """
        self._assert_type(ignore_params, list, "IGNORE_PARAMS")
        for t in ignore_params:
            self._assert_type(t, str, "IGNORE_PARAMS", "in ignored parameters")
        self._ignore_params = ignore_params

    @property
    def ARTIFACTS(self) -> List[Union[str, Dict[str, str]]]:
        """Get the artifacts to log for runs.
        Defaults to :attr:`["model_summary.txt", "metrics.csv",
        {"config.yaml": ".hydra"}]`.

        Returns:
            Artifacts to log for runs.
        """
        return self._artifacts

    @ARTIFACTS.setter
    def ARTIFACTS(self, artifacts: List[Union[str, Dict[str, str]]]) -> None:
        """Set the artifacts to log for runs.

        Args:
            artifacts: Artifacts to log for runs.

        Raises:
            ValueError: If the artifacts are not a list of strings or
                dictionaries.
        """
        self._assert_type(artifacts, list, "ARTIFACTS")
        for a in artifacts:
            if isinstance(a, dict):
                for k, v in a.items():
                    self._assert_type(k, str, "ARTIFACTS", "in artifacts")
                    self._assert_type(v, str, "ARTIFACTS", "in artifacts")
            else:
                self._assert_type(a, str, "ARTIFACTS", "in artifacts")
        self._artifacts = artifacts

from typing import List

from .abstract_constants import AbstractConstants


class TrainingConstants(AbstractConstants):
    """Singleton for managing the training configurations of `autrainer`."""

    _name = "TrainingConstants"
    _tasks = [
        "classification",
        "ml-classification",
        "regression",
        "mt-regression",
    ]

    @property
    def TASKS(self) -> List[str]:
        """Get the supported training tasks.
        Defaults to :attr:`["classification", "ml-classification",
        "regression", "mt-regression"]`.

        Returns:
            Supported training tasks.
        """
        return self._tasks

    @TASKS.setter
    def TASKS(self, tasks: List[str]):
        """Set the supported training tasks.

        Args:
            training_tasks: Supported training tasks.

        Raises:
            ValueError: If the training tasks are not a list of strings.
        """

        self._assert_type(tasks, list, "TASKS")
        for t in tasks:
            self._assert_type(t, str, "TASKS", "in training tasks")
        self._tasks = tasks

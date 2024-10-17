from typing import List

from .abstract_constants import AbstractConstants


class TrainingConstants(AbstractConstants):
    """Singleton for managing the training configurations of `autrainer`."""

    _tasks = ["classification", "regression", "ml-classification"]

    @property
    def TASKS(self) -> List[str]:
        """Get the supported training tasks.
        Defaults to :attr:`["classification", "regression",
        "ml-classification"]`.

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

        self._assert_type(tasks, list)
        for t in tasks:
            self._assert_type(t, str, "in training tasks")
        self._tasks = tasks

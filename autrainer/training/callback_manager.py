from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional


if TYPE_CHECKING:
    from .training import ModularTaskTrainer  # pragma: no cover


CALLBACK_FUNCTIONS = [
    # training loop callbacks
    "cb_on_train_begin",
    "cb_on_train_end",
    "cb_on_iteration_begin",
    "cb_on_iteration_end",
    "cb_on_step_begin",
    "cb_on_step_end",
    "cb_on_loader_exhausted",
    # validation loop callbacks
    "cb_on_val_begin",
    "cb_on_val_end",
    "cb_on_val_step_begin",
    "cb_on_val_step_end",
    # test loop callbacks
    "cb_on_test_begin",
    "cb_on_test_end",
    "cb_on_test_step_begin",
    "cb_on_test_step_end",
]


class CallbackSignature:
    @abstractmethod
    def cb_on_train_begin(self, trainer: "ModularTaskTrainer") -> None:
        """Called at the beginning of the training loop before the first
        iteration.

        Args:
            trainer: Mutable reference to the trainer.
        """

    @abstractmethod
    def cb_on_train_end(self, trainer: "ModularTaskTrainer") -> None:
        """Called at the end of the training loop after the last iteration,
        validation, and testing are completed.

        Args:
            trainer: Mutable reference to the trainer.
        """

    @abstractmethod
    def cb_on_iteration_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
    ) -> None:
        """Called at the beginning of each iteration.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training,
                this is the epoch number. For step-based training, this is the
                step number.
        """

    @abstractmethod
    def cb_on_iteration_end(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        metrics: dict,
    ) -> None:
        """Called at the end of each iteration including validation.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training,
                this is the epoch number. For step-based training, this is the
                step number.
            metrics: Dictionary of various metrics collected during the
                iteration.
        """

    @abstractmethod
    def cb_on_step_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        batch_idx: int,
    ) -> None:
        """Called at the beginning of step within an iteration.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training,
                this is the epoch number. For step-based training, this is the
                step number.
            batch_idx: Current batch index within the iteration. For
                epoch-based training, this is the batch index within the epoch.
                For step-based training, this is the step number modulo the
                evaluation frequency.
        """

    @abstractmethod
    def cb_on_step_end(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        batch_idx: int,
        loss: float,
    ) -> None:
        """Called at the end of step within an iteration.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training,
                this is the epoch number. For step-based training, this is the
                step number.
            batch_idx: Current batch index within the iteration. For
                epoch-based training, this is the batch index within the epoch.
                For step-based training, this is the step number modulo the
                evaluation frequency.
            loss: Reduced loss value for the batch.
        """

    @abstractmethod
    def cb_on_loader_exhausted(
        self, trainer: "ModularTaskTrainer", iteration: int
    ) -> None:
        """Called when the training data loader is exhausted.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training,
                this is the epoch number. For step-based training, this is the
                step number.
        """

    @abstractmethod
    def cb_on_val_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
    ) -> None:
        """Called at the beginning of the validation loop.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training,
                this is the epoch number. For step-based training, this is the
                step number.
        """

    @abstractmethod
    def cb_on_val_end(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        val_results: dict,
    ) -> None:
        """Called at the end of the validation loop.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training,
                this is the epoch number. For step-based training, this is the
                step number.
            val_results: Dictionary of validation results for the entire
                validation loop of the current iteration.
        """

    @abstractmethod
    def cb_on_val_step_begin(
        self, trainer: "ModularTaskTrainer", batch_idx: int
    ) -> None:
        """Called at the beginning of the validation step.

        Args:
            trainer: Mutable reference to the trainer.
            batch_idx: Current batch index within the validation loop.
        """

    @abstractmethod
    def cb_on_val_step_end(
        self,
        trainer: "ModularTaskTrainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        """Called at the end of the validation step.

        Args:
            trainer: Mutable reference to the trainer.
            batch_idx: Current batch index within the validation loop.
            loss: Reduced loss value for the batch.
        """

    @abstractmethod
    def cb_on_test_begin(self, trainer: "ModularTaskTrainer") -> None:
        """Called at the beginning of the testing loop.

        Args:
            trainer: Mutable reference to the trainer.
        """

    @abstractmethod
    def cb_on_test_end(
        self,
        trainer: "ModularTaskTrainer",
        test_results: dict,
    ) -> None:
        """Called at the end of the testing loop.

        Args:
            trainer: Mutable reference to the trainer.
            test_results: Dictionary of test results for the entire testing
                loop.
        """

    @abstractmethod
    def cb_on_test_step_begin(
        self,
        trainer: "ModularTaskTrainer",
        batch_idx: int,
    ) -> None:
        """Called at the beginning of the testing step.

        Args:
            trainer: Mutable reference to the trainer.
            batch_idx: Current batch index within the testing loop.
        """

    @abstractmethod
    def cb_on_test_step_end(
        self,
        trainer: "ModularTaskTrainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        """

        Args:
            trainer: Mutable reference to the trainer.
            batch_idx: Current batch index within the testing loop.
            loss: Reduced loss value for the batch.
        """


class CallbackManager:
    def __init__(self):
        self.callbacks = {cb: [] for cb in CALLBACK_FUNCTIONS}

    def register(self, obj: Optional[object] = None) -> None:
        if not obj:
            return
        for callback_name in CALLBACK_FUNCTIONS:
            obj_cb_function = getattr(obj, callback_name, None)
            if not obj_cb_function:
                continue
            self.callbacks[callback_name].append(obj_cb_function)

    def register_multiple(self, objs: List[Optional[object]]) -> None:
        for obj in objs:
            self.register(obj)

    def callback(self, position: str, **kwargs) -> None:
        if position not in self.callbacks:
            raise ValueError(f"Callback position {position} not found.")
        for cb in self.callbacks[position]:
            cb(**kwargs)

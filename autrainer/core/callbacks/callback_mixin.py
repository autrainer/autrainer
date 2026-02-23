from typing import TYPE_CHECKING, Any, Callable, Dict

from .callback_manager import CallbackManager


if TYPE_CHECKING:  # pragma: no cover
    from autrainer.training import ModularTaskTrainer


class CallbackMixin:
    def __init_subclass__(cls, **kwargs: Dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)
        names, seen = [], set()

        for c in cls.mro():
            if not issubclass(c, CallbackMixin):
                continue
            for name, attr in getattr(c, "__dict__", {}).items():
                if not (name.startswith("cb_") and callable(attr)):
                    continue
                if getattr(CallbackMixin, name, None) is attr:
                    continue
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)

        cls.__cb_names__ = names

    def __init__(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        mgr = CallbackManager()

        for name in type(self).__cb_names__:
            funcs = self._collect_chain_funcs(name)
            for func in reversed(funcs):  # base first
                bound = func.__get__(self, type(self))
                order = getattr(func, "__cb_order__", 0)
                mgr._register_bound(name, bound, order=order)

    def _collect_chain_funcs(self, name: str) -> list[Any]:
        impls = []
        for c in type(self).mro():
            if not issubclass(c, CallbackMixin):
                continue
            func = getattr(c, "__dict__", {}).get(name)
            if func is None or not callable(func):
                continue
            if getattr(CallbackMixin, name, None) is func:
                continue  # skip stub
            impls.append(func)

        if not impls:
            return []

        chain = [impls[0]]  # always include top level impl
        i = 0
        while i < len(impls) - 1 and getattr(chain[-1], "__cb_chain__", False):
            chain.append(impls[i + 1])
            i += 1

        return chain

    @staticmethod
    def order(order: int) -> Callable[..., Any]:
        """Decorator to set the order of the callback in the callback list.

        A larger order means the callback will be called later in the list.
        If multiple callbacks share the same order, they are applied in the order they
        were registered (i.e., in instantiation order of the objects).

        Note: If used in combination with CallbackMixin.chain, the order is determined
        by the order of the methods in the class hierarchy, with the base class method
        being registered first by default.

        Args:
            order: The order of the callback in the callback list. Defaults to 0.
        """

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            fn.__cb_order__ = order
            return fn

        return deco

    @staticmethod
    def chain() -> Callable[..., Any]:
        """Decorator to indicate that the callback should be chained and the base class
        implementation should also be called.

        By default, the base class implementation is called first, followed by the
        derived (decorated) implementation.
        """

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            fn.__cb_chain__ = True
            return fn

        return deco

    def cb_on_train_begin(self, trainer: "ModularTaskTrainer") -> None:
        """Called at the beginning of the training loop before the first iteration.

        Args:
            trainer: Mutable reference to the trainer.
        """

    def cb_on_train_end(self, trainer: "ModularTaskTrainer") -> None:
        """Called at the end of the training loop after the last iteration, validation,
        and testing are completed.

        Args:
            trainer: Mutable reference to the trainer.
        """

    def cb_on_iteration_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
    ) -> None:
        """Called at the beginning of each iteration.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training, this is the
                epoch number. For step-based training, this is the step number.
        """

    def cb_on_iteration_end(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        metrics: Dict[str, float],
    ) -> None:
        """Called at the end of each iteration including validation.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training, this is the
                epoch number. For step-based training, this is the step number.
            metrics: Dictionary of various metrics collected during the iteration.
        """

    def cb_on_step_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        batch_idx: int,
    ) -> None:
        """Called at the beginning of step within an iteration.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training, this is the
                epoch number. For step-based training, this is the step number.
            batch_idx: Current batch index within the iteration. For epoch-based
                training, this is the batch index within the epoch. For step-based
                training, this is the step number modulo the evaluation frequency.
        """

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
            iteration: Current iteration number. For epoch-based training, this is the
                epoch number. For step-based training, this is the step number.
            batch_idx: Current batch index within the iteration. For epoch-based
                training, this is the batch index within the epoch. For step-based
                training, this is the step number modulo the evaluation frequency.
            loss: Reduced loss value for the batch.
        """

    def cb_on_loader_exhausted(
        self, trainer: "ModularTaskTrainer", iteration: int
    ) -> None:
        """Called when the training data loader is exhausted.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training, this is the
                epoch number. For step-based training, this is the step number.
        """

    def cb_on_dev_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
    ) -> None:
        """Called at the beginning of the validation loop.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training, this is the
                epoch number. For step-based training, this is the step number.
        """

    def cb_on_dev_end(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        dev_results: Dict[str, float],
    ) -> None:
        """Called at the end of the validation loop.

        Args:
            trainer: Mutable reference to the trainer.
            iteration: Current iteration number. For epoch-based training, this is the
                epoch number. For step-based training, this is the step number.
            dev_results: Dictionary of validation results for the entire validation loop
                of the current iteration.
        """

    def cb_on_dev_step_begin(
        self,
        trainer: "ModularTaskTrainer",
        batch_idx: int,
    ) -> None:
        """Called at the beginning of the validation step.

        Args:
            trainer: Mutable reference to the trainer.
            batch_idx: Current batch index within the validation loop.
        """

    def cb_on_dev_step_end(
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

    def cb_on_test_begin(self, trainer: "ModularTaskTrainer") -> None:
        """Called at the beginning of the testing loop.

        Args:
            trainer: Mutable reference to the trainer.
        """

    def cb_on_test_end(
        self,
        trainer: "ModularTaskTrainer",
        test_results: Dict[str, float],
    ) -> None:
        """Called at the end of the testing loop.

        Args:
            trainer: Mutable reference to the trainer.
            test_results: Dictionary of test results for the entire testing loop.
        """

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

    def cb_on_test_step_end(
        self,
        trainer: "ModularTaskTrainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        """Called at the end of the testing step.

        Args:
            trainer: Mutable reference to the trainer.
            batch_idx: Current batch index within the testing loop.
            loss: Reduced loss value for the batch.
        """

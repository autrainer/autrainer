from collections import defaultdict
import inspect
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Dict, List, Tuple
import weakref


if TYPE_CHECKING:  # pragma: no cover
    from autrainer.training import ModularTaskTrainer


class CallbackManager:
    _instance: "CallbackManager" = None
    _receivers: DefaultDict[str, List[Tuple[int, weakref.WeakMethod]]]

    def __new__(cls) -> "CallbackManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._receivers = defaultdict(list)
        return cls._instance

    def _remove_dead_receivers(self, name: str) -> None:
        alive: List[Tuple[int, weakref.WeakMethod]] = []
        if name not in self._receivers:
            return
        for o, w in self._receivers[name]:
            if w() is not None:
                alive.append((o, w))
        self._receivers[name] = alive

    def _register_bound(self, name: str, fn: Callable[..., Any], order: int) -> None:
        self._remove_dead_receivers(name)
        self._receivers[name].append((order, weakref.WeakMethod(fn)))
        self._receivers[name].sort(key=lambda t: t[0])

    def _emit(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        name = inspect.currentframe().f_back.f_code.co_name
        self._remove_dead_receivers(name)
        for _, w in self._receivers.get(name, []):
            if (fn := w()) is None:
                continue
            fn(*args, **kwargs)

    def remove(self, obj: object, name: str) -> None:
        """Remove a callback receiver for a specific callback name.

        Args:
            obj: Receiver object whose callback should be removed.
            name: Name of the callback from which the receiver should be removed.
        """
        alive: List[weakref.WeakMethod] = []
        for o, w in self._receivers.get(name, []):
            if (fn := w()) is None or fn.__self__ is obj:
                continue
            alive.append((o, w))
        self._receivers[name] = alive

    def remove_all(self, obj: object) -> None:
        """Remove all callback receivers associated with the given object across all
        callbacks.

        Args:
            obj: Receiver object whose callbacks should be removed.
        """
        for name in list(self._receivers.keys()):
            self.remove(obj, name)

    def cb_on_train_begin(self, trainer: "ModularTaskTrainer") -> None:
        self._emit(trainer=trainer)

    def cb_on_train_end(self, trainer: "ModularTaskTrainer") -> None:
        self._emit(trainer=trainer)

    def cb_on_iteration_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
    ) -> None:
        self._emit(trainer=trainer, iteration=iteration)

    def cb_on_iteration_end(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        metrics: Dict[str, float],
    ) -> None:
        self._emit(trainer=trainer, iteration=iteration, metrics=metrics)

    def cb_on_step_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        batch_idx: int,
    ) -> None:
        self._emit(trainer=trainer, iteration=iteration, batch_idx=batch_idx)

    def cb_on_step_end(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        batch_idx: int,
        loss: float,
    ) -> None:
        self._emit(trainer=trainer, iteration=iteration, batch_idx=batch_idx, loss=loss)

    def cb_on_loader_exhausted(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
    ) -> None:
        self._emit(trainer=trainer, iteration=iteration)

    def cb_on_dev_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
    ) -> None:
        self._emit(trainer=trainer, iteration=iteration)

    def cb_on_dev_end(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
        dev_results: Dict[str, float],
    ) -> None:
        self._emit(trainer=trainer, iteration=iteration, dev_results=dev_results)

    def cb_on_dev_step_begin(
        self,
        trainer: "ModularTaskTrainer",
        batch_idx: int,
    ) -> None:
        self._emit(trainer=trainer, batch_idx=batch_idx)

    def cb_on_dev_step_end(
        self,
        trainer: "ModularTaskTrainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        self._emit(trainer=trainer, batch_idx=batch_idx, loss=loss)

    def cb_on_test_begin(self, trainer: "ModularTaskTrainer") -> None:
        self._emit(trainer=trainer)

    def cb_on_test_end(
        self,
        trainer: "ModularTaskTrainer",
        test_results: Dict[str, float],
    ) -> None:
        self._emit(trainer=trainer, test_results=test_results)

    def cb_on_test_step_begin(
        self,
        trainer: "ModularTaskTrainer",
        batch_idx: int,
    ) -> None:
        self._emit(trainer=trainer, batch_idx=batch_idx)

    def cb_on_test_step_end(
        self,
        trainer: "ModularTaskTrainer",
        batch_idx: int,
        loss: float,
    ) -> None:
        self._emit(trainer=trainer, batch_idx=batch_idx, loss=loss)

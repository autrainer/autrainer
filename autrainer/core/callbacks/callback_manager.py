from __future__ import annotations

from collections import defaultdict
import inspect
from typing import TYPE_CHECKING, Any
import weakref


if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from autrainer.training import ModularTaskTrainer


class CallbackManager:
    _instance: CallbackManager | None = None
    _receivers: defaultdict[str, list[tuple[int, weakref.WeakMethod]]]

    def __new__(cls) -> "CallbackManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._receivers = defaultdict(list)
        return cls._instance

    def _remove_dead_receivers(self, name: str) -> None:
        alive: list[tuple[int, weakref.WeakMethod]] = []
        if name not in self._receivers:
            return
        for o, w in self._receivers[name]:
            if w() is not None:
                alive.append((o, w))
        self._receivers[name] = alive

    def _register(self, name: str, fn: Callable[..., Any], order: int) -> None:
        self._remove_dead_receivers(name)
        self._receivers[name].append((order, weakref.WeakMethod(fn)))
        self._receivers[name].sort(key=lambda t: t[0])

    def _emit(self, *args: Any, **kwargs: Any) -> None:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            return

        name = frame.f_back.f_code.co_name
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
        alive: list[tuple[int, weakref.WeakMethod]] = []
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
        metrics: dict[str, float],
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
        dev_results: dict[str, float],
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
        test_results: dict[str, float],
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

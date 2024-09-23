from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from autrainer.training import ModularTaskTrainer


class LRTrackerCallback:
    def cb_on_train_begin(self, trainer: "ModularTaskTrainer") -> None:
        self.lr = trainer.optimizer.param_groups[0]["lr"]

    def cb_on_iteration_begin(
        self,
        trainer: "ModularTaskTrainer",
        iteration: int,
    ) -> None:
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        if current_lr != self.lr:
            print(
                f"Learning rate changed from {self.lr} "
                f"to {current_lr} in iteration {iteration}."
            )
            self.lr = current_lr

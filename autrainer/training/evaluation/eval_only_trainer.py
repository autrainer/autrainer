import os

from omegaconf import DictConfig, OmegaConf
import torch

from autrainer.training.continue_training import ContinueTraining

from ..utils import load_pretrained_model_state
from .evaluator import Evaluator


class EvalOnlyTrainer:
    def __init__(self, cfg: DictConfig, output_directory: str) -> None:
        from ..trainer import Trainer  # avoid circular import

        cfg["dataset"] = cfg.pop("evaluation")
        self.base_run_directory = self._get_base_run_dir(output_directory)

        exp_id = os.path.basename(os.path.dirname(os.path.dirname(output_directory)))
        self.trainer = Trainer(
            cfg,
            output_directory,
            experiment_id=f"{exp_id}/eval",
            save_initial_states=False,
        )
        self.cp = cfg.checkpoint

        self.evaluator = Evaluator()

        # remove continue training from callback as invalid for eval
        for k, cbs in self.trainer.callback_manager.callbacks.items():
            self.trainer.callback_manager.callbacks[k] = [
                cb
                for cb in cbs
                if not isinstance(getattr(cb, "__self__", None), ContinueTraining)
            ]

        sd = torch.load(
            os.path.join(
                self.base_run_directory,
                f"{cfg.training_type}_{cfg.checkpoint}",
                "model.pt",
            ),
            map_location="cpu",
            weights_only=True,
        )
        load_pretrained_model_state(self.trainer.model, sd, skip_last_layer=False)

    @staticmethod
    def _get_base_run_dir(output_directory: str) -> str:
        training_dir = os.path.join(
            os.path.dirname(os.path.dirname(output_directory)),
            "training",
        )
        run = os.path.basename(output_directory).split("_", 2)[-1]
        return os.path.join(training_dir, run)

    def eval(self) -> None:
        self.trainer.model = self.trainer.model.to(self.trainer.DEVICE)
        self.trainer.callback_manager.callback(
            position="cb_on_train_begin",
            trainer=self.trainer,
        )

        self.evaluator.dev(self.trainer, self.cp, track_best=False)
        self.evaluator.test(self.trainer)

        self.trainer.bookkeeping.save_results_df(self.trainer.metrics, "metrics.csv")
        self.trainer.callback_manager.callback(
            position="cb_on_train_end",
            trainer=self.trainer,
        )
        OmegaConf.save(
            {
                "train": self.trainer.train_timer.save(),
                "dev": self.trainer.dev_timer.save(),
                "test": self.trainer.test_timer.save(),
            },
            os.path.join(self.trainer.output_directory, "timers.yaml"),
        )
        return self.trainer.metrics.loc[self.cp][self.trainer.data.tracking_metric.name]

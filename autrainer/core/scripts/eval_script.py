import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
import yaml

import autrainer

from .abstract_script import AbstractScript, MockParser
from .command_line_error import CommandLineError
from .utils import (
    add_hydra_args_to_sys,
    catch_cli_errors,
    check_invalid_config_path_arg,
    run_hydra_cmd,
    running_in_notebook,
)


class EvalScript(AbstractScript):
    def __init__(self) -> None:
        super().__init__(
            "eval",
            "Launch an evaluation configuration (Hydra).",
            extended_description=(
                "For more information on Hydra's command line line flags, see:\n"
                "https://hydra.cc/docs/advanced/hydra-command-line-flags/."
            ),
            epilog="Example: autrainer eval -cn eval.yaml",
            unknown_args=True,
        )

    def _assert_valid_checkpoint(self, cfg: DictConfig) -> None:
        valid = {"best", "last"}
        if isinstance(cfg.checkpoint, int) or cfg.checkpoint in valid:
            return
        msg = f"Checkpoint '{cfg.checkpoint}' must be an iteration or in '{valid}'."
        raise CommandLineError(self.parser, msg, code=1)

    def _get_base_run_dir(self, output_dir: str) -> str:
        training_dir = os.path.join(
            os.path.dirname(os.path.dirname(output_dir)),
            "training",
        )
        run = os.path.basename(output_dir).split("_", 2)[-1]
        return os.path.join(training_dir, run)

    def _create_cp(self, cfg: DictConfig, output_dir: str) -> int:
        def _find_cp(cfg: DictConfig, base_run: str) -> Optional[int]:
            if cfg.checkpoint == "best":
                if not os.path.exists(os.path.join(base_run, "_best", "dev.yaml")):
                    return None
                with open(os.path.join(base_run, "_best", "dev.yaml")) as f:
                    return int(yaml.safe_load(f)["iteration"])
            if cfg.checkpoint == "last":
                cps = os.listdir(base_run)
                if not any(d.startswith(f"{cfg.training_type}_") for d in cps):
                    return None
                return max(
                    int(d.split("_")[-1])
                    for d in os.listdir(base_run)
                    if d.startswith(f"{cfg.training_type}_")
                )
            return int(cfg.checkpoint)

        base_run = self._get_base_run_dir(output_dir)
        cp = _find_cp(cfg, base_run)
        if cp is None:
            run = os.path.basename(base_run)
            msg = f"Checkpoint '{cfg.checkpoint}' does not exist for run '{run}'."
            raise CommandLineError(self.parser, msg, code=1)
        return cp

    def _assert_base_state_exists(self, cfg: DictConfig, output_dir: str) -> None:
        cp = self._create_cp(cfg, output_dir)
        base_run = self._get_base_run_dir(output_dir)
        if os.path.exists(
            os.path.join(base_run, f"{cfg.training_type}_{cp}", "model.pt")
        ):
            return
        run = os.path.basename(base_run)
        msg = f"Checkpoint '{cp}' does not exist for run '{run}'."
        raise CommandLineError(self.parser, msg, code=1)

    def main(self, args: dict) -> None:
        @autrainer.main("eval")
        def main(cfg: DictConfig) -> float:
            import hydra

            from autrainer.core.filters import AlreadyRun

            OmegaConf.set_struct(cfg, False)
            OmegaConf.resolve(cfg)
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

            # ? Skip if run exists and return best tracking metric
            if AlreadyRun(cfg, output_dir).filter():
                import autrainer
                from autrainer.core.utils import Bookkeeping
                from autrainer.metrics import AbstractMetric

                cp = self._create_cp(cfg, output_dir)
                dev_metrics = OmegaConf.load(
                    os.path.join(output_dir, f"{cfg.training_type}_{cp}", "dev.yaml")
                )
                tracking_metric = autrainer.instantiate_shorthand(
                    config=cfg.evaluation.tracking_metric,
                    instance_of=AbstractMetric,
                )
                best_metric = dev_metrics[tracking_metric.name]["all"]

                bookkeeping = Bookkeeping(output_dir)
                bookkeeping.log(f"Skipping: {os.path.basename(output_dir)}")

                return best_metric

            self._assert_valid_checkpoint(cfg)
            self._assert_base_state_exists(cfg, output_dir)

            # ? Save cfg to output directory
            cfg_path = os.path.join(output_dir, ".hydra", "config.yaml")
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            OmegaConf.save(cfg, cfg_path)

            from autrainer.training.evaluation import EvalOnlyTrainer

            cfg.checkpoint = self._create_cp(cfg, output_dir)  # patch checkpoint
            eval_trainer = EvalOnlyTrainer(cfg, output_dir)
            return eval_trainer.eval()

        check_invalid_config_path_arg(self.parser)
        main()


@catch_cli_errors
def eval(
    override_kwargs: Optional[Dict[str, Any]] = None,
    config_name: str = "eval",
    config_path: Optional[str] = None,
) -> None:
    """Launch an evaluation configuration.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            evaluation script. Defaults to None.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "eval".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    if running_in_notebook():
        run_hydra_cmd("eval", override_kwargs, config_name, config_path)
    else:
        add_hydra_args_to_sys(override_kwargs, config_name, config_path)
        script = EvalScript()
        script.parser = MockParser()
        script.main({})

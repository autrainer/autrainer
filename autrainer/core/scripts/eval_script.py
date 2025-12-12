import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

import autrainer

from .abstract_script import AbstractScript, MockParser
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

                dev_metrics = OmegaConf.load(
                    os.path.join(output_dir, "_best", "dev.yaml")
                )
                tracking_metric = autrainer.instantiate_shorthand(
                    config=cfg.evaluation.tracking_metric,
                    instance_of=AbstractMetric,
                )
                best_metric = dev_metrics[tracking_metric.name]["all"]

                bookkeeping = Bookkeeping(output_dir)
                bookkeeping.log(f"Skipping: {os.path.basename(output_dir)}")

                return best_metric

            # ? Save cfg to output directory
            cfg_path = os.path.join(output_dir, ".hydra", "config.yaml")
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            OmegaConf.save(cfg, cfg_path)

            from autrainer.training.evaluation import EvalOnlyTrainer

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

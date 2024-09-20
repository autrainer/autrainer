from typing import Optional

from omegaconf import DictConfig, OmegaConf

import autrainer

from .abstract_script import AbstractScript, MockParser
from .utils import (
    add_hydra_args_to_sys,
    catch_cli_errors,
    run_hydra_cmd,
    running_in_notebook,
)


class TrainScript(AbstractScript):
    def __init__(self) -> None:
        super().__init__(
            "train",
            "Launch a training configuration (Hydra).",
            extended_description=(
                "For more information on Hydra's command line line flags, see:\n"
                "https://hydra.cc/docs/advanced/hydra-command-line-flags/."
            ),
            epilog="Example: autrainer train -cn config.yaml",
            unknown_args=True,
        )

    def main(self, args: dict) -> None:
        @autrainer.main("config")
        def main(cfg: DictConfig) -> float:
            import os

            import hydra

            OmegaConf.set_struct(cfg, False)
            OmegaConf.resolve(cfg)
            output_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )

            # ? Skip if run exists and return best tracking metric
            if os.path.exists(os.path.join(output_dir, "metrics.csv")):
                import autrainer
                from autrainer.core.utils import Bookkeeping
                from autrainer.metrics import AbstractMetric

                dev_metrics = OmegaConf.load(
                    os.path.join(output_dir, "_best", "dev.yaml")
                )
                tracking_metric = autrainer.instantiate_shorthand(
                    config=cfg.dataset.tracking_metric,
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

            from autrainer.training import ModularTaskTrainer

            trainer = ModularTaskTrainer(
                cfg=cfg,
                output_directory=output_dir,
            )
            return trainer.train()

        main()


@catch_cli_errors
def train(
    override_kwargs: Optional[dict] = None,
    config_name: str = "config",
    config_path: Optional[str] = None,
) -> None:
    """Launch a training configuration.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "config".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    if running_in_notebook():
        run_hydra_cmd("train", override_kwargs, config_name, config_path)
    else:
        add_hydra_args_to_sys(override_kwargs, config_name, config_path)
        script = TrainScript()
        script.parser = MockParser()
        script.main({})

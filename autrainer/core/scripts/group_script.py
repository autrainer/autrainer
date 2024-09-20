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


class GroupScript(AbstractScript):
    def __init__(self) -> None:
        super().__init__(
            "group",
            "Launch a manual grouping of multiple grid search results (Hydra).",
            extended_description=(
                "For more information on Hydra's command line line flags, see:\n"
                "https://hydra.cc/docs/advanced/hydra-command-line-flags/."
            ),
            epilog="Example: autrainer group -cn group.yaml",
            unknown_args=True,
        )

    def main(
        self,
        args: dict,
        config_name: str = "group",
        config_path: Optional[str] = None,
    ) -> None:
        @autrainer.main(config_name=config_name, config_path=config_path)
        def main(cfg: DictConfig) -> None:
            from autrainer.postprocessing import GroupGrid

            OmegaConf.set_struct(cfg, False)
            OmegaConf.resolve(cfg)
            ma = GroupGrid(
                results_dir=cfg.results_dir,
                groupings=cfg.groupings,
                max_runs=cfg.max_runs,
                plot_params=cfg.plotting,
            )

            ma.group_runs()

        main()


@catch_cli_errors
def group(
    override_kwargs: Optional[dict] = None,
    config_name: str = "group",
    config_path: Optional[str] = None,
) -> None:
    """Launch a manual grouping of multiple grid search results.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "group".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    if running_in_notebook():
        run_hydra_cmd("group", override_kwargs, config_name, config_path)
    else:
        add_hydra_args_to_sys(override_kwargs, config_name, config_path)
        script = GroupScript()
        script.parser = MockParser()
        script.main({})

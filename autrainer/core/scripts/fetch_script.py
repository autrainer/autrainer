from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import autrainer
from autrainer.core.scripts.abstract_script import MockParser

from .abstract_preprocess_script import AbstractPreprocessScript, PreprocessArgs
from .utils import (
    add_hydra_args_to_sys,
    catch_cli_errors,
    check_invalid_config_path_arg,
    run_hydra_cmd,
    running_in_notebook,
)


class FetchScript(AbstractPreprocessScript):
    def __init__(self) -> None:
        super().__init__(
            "fetch",
            (
                "Fetch the datasets, models, and transforms specified in a "
                "training configuration (Hydra)."
            ),
            extended_description=(
                "For more information on Hydra's command line line flags, see:\n"
                "https://hydra.cc/docs/advanced/hydra-command-line-flags/."
            ),
            epilog="Example: autrainer fetch -cn config.yaml",
            dataclass=PreprocessArgs,
            unknown_args=True,
        )

    def main(self, args: PreprocessArgs) -> None:
        self._override_launcher(args)
        self.datasets = {}
        self.models = {}

        @autrainer.main("config")
        def main(cfg: DictConfig) -> None:
            if not self._id_in_dict(self.datasets, cfg.dataset.id):
                self.datasets[cfg.dataset.id] = OmegaConf.to_container(cfg.dataset)

            if not self._id_in_dict(self.models, cfg.model.id):
                self.models[cfg.model.id] = OmegaConf.to_container(cfg.model)

        check_invalid_config_path_arg(self.parser)
        main()
        self._download_datasets()
        self._download_models()
        self._clean_up()

    def _download_datasets(self) -> None:
        print("Fetching datasets...")
        for name, dataset in self.datasets.items():
            print(f" - {name}")
            if not dataset.get("path", None):
                continue
            hydra.utils.instantiate(
                config={
                    "_target_": dataset["_target_"] + ".download",
                    "path": dataset["path"],
                },
            )  # pragma: no cover
            self._download_transform(dataset.get("transform", None))

    def _download_models(self) -> None:
        print("Fetching models...")
        for name, model in self.models.items():
            print(f" - {name}")
            model_checkpoint = model.pop("model_checkpoint", None)
            if model_checkpoint:
                continue  # no need to download if checkpoint is provided
            self._download_transform(model.pop("transform", None))
            model.pop("optimizer_checkpoint", None)
            model.pop("scheduler_checkpoint", None)
            model.pop("skip_last_layer", None)
            autrainer.instantiate(config=model, output_dim=10)

    def _download_transform(self, transform: Optional[DictConfig]) -> None:
        if not transform:
            return
        from autrainer.transforms import TransformManager

        for subset in ["base", "train", "dev", "test"]:
            if (transforms := transform.get(subset, None)) is None:
                continue
            for t in transforms:
                if isinstance(t, dict) and next(iter(t.values())) is None:
                    continue  # skip removed transforms
                TransformManager._instantiate(t)


@catch_cli_errors
def fetch(
    override_kwargs: Optional[dict] = None,
    cfg_launcher: bool = False,
    config_name: str = "config",
    config_path: Optional[str] = None,
) -> None:
    """Fetch the datasets, models, and transforms specified in a training
    configuration.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        cfg_launcher: Use the launcher specified in the configuration instead
            of the Hydra basic launcher. Defaults to False.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "config".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    if running_in_notebook():
        run_hydra_cmd(
            "fetch -l" if cfg_launcher else "fetch",
            override_kwargs,
            config_name,
            config_path,
        )

    else:
        add_hydra_args_to_sys(override_kwargs, config_name, config_path)
        script = FetchScript()
        script.parser = MockParser()
        script.main(PreprocessArgs(cfg_launcher))

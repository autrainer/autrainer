from dataclasses import dataclass
import torch
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf

import autrainer
from autrainer.core.scripts.abstract_script import MockParser

from .abstract_preprocess_script import (
    AbstractPreprocessScript,
    PreprocessArgs,
)
from .utils import (
    add_hydra_args_to_sys,
    catch_cli_errors,
    run_hydra_cmd,
    running_in_notebook,
)


@dataclass
class PreprocessArgs(PreprocessArgs):
    num_workers: int
    update_frequency: int


class PreprocessScript(AbstractPreprocessScript):
    def __init__(self) -> None:
        super().__init__(
            "preprocess",
            "Launch a data preprocessing configuration (Hydra).",
            extended_description=(
                "For more information on Hydra's command line line flags, see:\n"
                "https://hydra.cc/docs/advanced/hydra-command-line-flags/."
            ),
            epilog="Example: autrainer preprocess -cn config.yaml",
            dataclass=PreprocessArgs,
            unknown_args=True,
        )

    def add_arguments(self) -> None:
        super().add_arguments()
        self.parser.add_argument(
            "-n",
            "--num-workers",
            type=int,
            default=1,
            metavar="N",
            required=False,
            help=(
                "Number of workers (threads) to use for preprocessing. "
                "Defaults to 1."
            ),
        )
        self.parser.add_argument(
            "-u",
            "--update-frequency",
            type=int,
            default=1,
            metavar="F",
            required=False,
            help=(
                "Frequency of progress bar updates for each worker (thread). "
                "If 0, the progress bar will be disabled. Defaults to 1."
            ),
        )

    def main(self, args: PreprocessArgs) -> None:
        self._assert_num_workers(args.num_workers)

        import hydra

        self.num_workers = args.num_workers
        self.update_frequency = args.update_frequency
        self._override_launcher(args)
        self.datasets = {}
        self.preprocessing = {}

        @autrainer.main("config")
        def main(cfg: DictConfig) -> None:
            from hydra.errors import MissingConfigException

            if self._id_in_dict(self.datasets, cfg.dataset.id):
                return

            self.datasets[cfg.dataset.id] = OmegaConf.to_container(cfg.dataset)
            preprocessing_cfg = None
            if cfg.dataset.get("features_subdir"):
                try:
                    preprocessing_cfg = OmegaConf.to_container(
                        hydra.compose(
                            f"preprocessing/{cfg.dataset.features_subdir}"
                        )["preprocessing"]
                    )
                except MissingConfigException:
                    pass
            self.preprocessing[cfg.dataset.id] = preprocessing_cfg

        main()
        self._preprocess_datasets()
        self._clean_up()

    def _assert_num_workers(self, num_workers: int) -> None:
        if num_workers < 1:
            raise ValueError(
                f"Number of workers '{num_workers}' must be >= 1."
            )

    def _preprocess_datasets(self) -> None:
        import os
        from pathlib import Path

        import pandas as pd
        from tqdm import tqdm

        from autrainer.datasets.utils import AbstractFileHandler
        from autrainer.transforms import SmartCompose

        print("Preprocessing datasets...")
        for (name, dataset), preprocess in zip(
            self.datasets.items(), self.preprocessing.values()
        ):
            print(f" - {name}")
            if preprocess is None:
                print("No preprocessing specified. Skipping...")
                continue
            # swap dataset handler with preprocessing handler
            features_subdir = dataset["features_subdir"]
            output_file_handler = autrainer.instantiate_shorthand(
                config=dataset["file_handler"],
                instance_of=AbstractFileHandler,
            )
            output_file_type = dataset["file_type"]
            # set features_subdir to None (defaults to audio)
            # so that iteration takes place over raw data
            dataset["features_subdir"] = None
            dataset["file_handler"] = preprocess["file_handler"]
            dataset["file_type"] = preprocess["file_type"]
            data = autrainer.instantiate_shorthand(dataset)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(
                    data.train_dataset,
                    data.dev_dataset,
                    data.test_dataset
                ),
                shuffle=False,
                num_workers=self.num_workers,
                batch_size=1  #TODO: can we do it batched?
            )
            pipeline = SmartCompose(
                [
                    autrainer.instantiate_shorthand(t)
                    for t in preprocess["pipeline"]
                ]
            )
            for d, n in (
                (data.train_dataset, "train"),
                (data.dev_dataset, "dev"),
                (data.test_dataset, "test"),
            ):
                loader = torch.utils.data.DataLoader(
                    dataset=d,
                    shuffle=False,
                    num_workers=self.num_workers,
                    batch_size=1  #TODO: can we do it batched?
                )
                for data in tqdm.tqdm(
                    loader,
                    total=len(loader),
                    desc=f"{name}-{n}",
                    disable=self.update_frequency == 0
                ):
                    #TODO: will be streamlined once we switch to dataclass
                    index = d.df.index[data[2]]
                    item_path = d.df.loc[index, d.index_column]
                    out_path = Path(
                        dataset["path"],
                        features_subdir,
                        os.path.basename(item_path),
                    ).with_suffix("." + output_file_type)
                    if os.path.exists(out_path):
                        continue
                    output_file_handler.save(
                        out_path,
                        pipeline(data[0], 0)
                    )


@catch_cli_errors
def preprocess(
    override_kwargs: Optional[dict] = None,
    num_workers: int = 1,
    update_frequency: int = 1,
    cfg_launcher: bool = False,
    config_name: str = "config",
    config_path: Optional[str] = None,
) -> None:
    """Launch a data preprocessing configuration.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        num_workers: Number of workers (threads) to use for preprocessing.
            Defaults to 1.
        update_frequency: Frequency of progress bar updates for each worker
            (thread). If 0, the progress bar will be disabled. Defaults to 1.
        cfg_launcher: Use the launcher specified in the configuration instead
            of the Hydra basic launcher. Defaults to False.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "config".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    if running_in_notebook():
        cmd = "preprocess"
        if cfg_launcher:
            cmd += " -l"
        cmd += f" -n {num_workers} -u {update_frequency}"
        run_hydra_cmd(cmd, override_kwargs, config_name, config_path)

    else:
        add_hydra_args_to_sys(override_kwargs, config_name, config_path)
        script = PreprocessScript()
        script.parser = MockParser()
        script.main(
            PreprocessArgs(cfg_launcher, num_workers, update_frequency)
        )

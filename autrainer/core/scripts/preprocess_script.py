import contextlib
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class PreprocessArgs(PreprocessArgs):
    num_workers: int
    update_frequency: int


def preprocess_main(
    name: str,
    dataset: DictConfig,
    preprocess: DictConfig,
    num_workers: int,
    update_frequency: int,
) -> None:
    from pathlib import Path

    import torch
    from tqdm import tqdm

    from autrainer.core.structs import DataBatch
    from autrainer.datasets import AbstractDataset
    from autrainer.datasets.utils import AbstractFileHandler
    from autrainer.transforms import SmartCompose

    print(f" - {name}")
    if preprocess is None:
        print("No preprocessing specified. Skipping...")
        return
    # set dataset file handler as output handler
    # as dataset is already configured
    # to work with the output of the preprocessing script
    output_file_handler = autrainer.instantiate_shorthand(
        config=dataset["file_handler"],
        instance_of=AbstractFileHandler,
    )
    output_file_type = dataset["file_type"]

    # override dataset file handling to work with raw audio
    dataset["file_handler"] = preprocess["file_handler"]
    # None allows dataset to work with all audio files
    dataset["file_type"] = None
    dataset["seed"] = 0  # ignored
    dataset.pop("criterion")
    dataset.pop("transform")
    for subset in ["train", "dev", "test"]:
        dataset.pop(f"{subset}_loader_kwargs", None)
    features_path = dataset.pop("features_path", None)
    if features_path is None:
        features_path = dataset["path"]
    features_subdir = dataset["features_subdir"]

    dataset["features_subdir"] = None
    data = autrainer.instantiate(dataset, AbstractDataset)
    pipeline = SmartCompose(
        [autrainer.instantiate_shorthand(t) for t in preprocess["pipeline"]]
    )
    # use pipeline to process the features
    data.train_transform = pipeline
    data.dev_transform = pipeline
    data.test_transform = pipeline

    for d, df, n in (
        (data.train_dataset, data.df_train, "train"),
        (data.dev_dataset, data.df_dev, "dev"),
        (data.test_dataset, data.df_test, "test"),
    ):
        # TODO: dataloader underutilized
        # workers only parallelize loading
        loader = torch.utils.data.DataLoader(
            dataset=d,
            shuffle=False,
            num_workers=num_workers,
            batch_size=1,
            collate_fn=DataBatch.collate,
        )
        for instance in tqdm(
            loader,
            total=len(loader),
            desc=f"{name}-{n}",
            disable=update_frequency == 0,
        ):
            item_path = df.loc[df.index[int(instance.index)], d.index_column]
            out_path = Path(
                features_path,
                features_subdir,
                item_path,
            ).with_suffix(f".{output_file_type}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                continue
            output_file_handler.save(
                out_path,
                instance.features.squeeze(dim=0),
            )


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
            default=0,
            metavar="N",
            required=False,
            help=(
                "Number of workers (subprocesses) to use for preprocessing. "
                "Defaults to 0."
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
                "Frequency of progress bar updates for each worker (subprocess). "
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
                with contextlib.suppress(MissingConfigException):
                    preprocessing_cfg = OmegaConf.to_container(
                        hydra.compose(f"preprocessing/{cfg.dataset.features_subdir}")[
                            "preprocessing"
                        ]
                    )
            self.preprocessing[cfg.dataset.id] = preprocessing_cfg

        check_invalid_config_path_arg(self.parser)
        main()
        self._preprocess_datasets()
        self._clean_up()

    def _assert_num_workers(self, num_workers: int) -> None:
        if num_workers < 0:
            raise ValueError(f"Number of workers '{num_workers}' must be >= 0.")

    def _preprocess_datasets(self) -> None:
        print("Preprocessing datasets...")
        for (name, dataset), preprocess in zip(
            self.datasets.items(), self.preprocessing.values(), strict=False
        ):
            preprocess_main(
                name=name,
                dataset=dataset,
                preprocess=preprocess,
                num_workers=self.num_workers,
                update_frequency=self.update_frequency,
            )


@catch_cli_errors
def preprocess(
    override_kwargs: Optional[dict] = None,
    num_workers: int = 0,
    update_frequency: int = 1,
    cfg_launcher: bool = False,
    config_name: str = "config",
    config_path: Optional[str] = None,
) -> None:
    """Launch a data preprocessing configuration.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        num_workers: Number of workers (subprocess) to use for preprocessing.
            Defaults to 0.
        update_frequency: Frequency of progress bar updates.
            If 0, the progress bar will be disabled. Defaults to 1.
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
        script.main(PreprocessArgs(cfg_launcher, num_workers, update_frequency))

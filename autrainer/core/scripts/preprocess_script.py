from dataclasses import dataclass
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf

import autrainer

from .abstract_preprocess_script import (
    AbstractPreprocessScript,
    PreprocessArgs,
)
from .command_line_error import catch_cli_errors
from .utils import run_autrainer_hydra_cmd


@dataclass
class PreprocessArgs(PreprocessArgs):
    num_workers: int
    pbar_frequency: int
    silent: bool


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
            default=-1,
            metavar="N",
            required=False,
            help="Number of workers to use for preprocessing. Defaults to -1.",
        )
        self.parser.add_argument(
            "-p",
            "--pbar-frequency",
            type=int,
            default=100,
            metavar="P",
            required=False,
            help="Frequency of progress bar updates. Defaults to 100.",
        )
        self.parser.add_argument(
            "-s",
            "--silent",
            action="store_true",
            default=False,
            required=False,
            help="Disable progress bar output.",
        )

    def main(self, args: PreprocessArgs) -> None:
        import hydra

        self.num_workers = args.num_workers
        self.pbar_frequency = args.pbar_frequency
        self.silent = args.silent
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

    def _preprocess_datasets(self) -> None:
        from concurrent.futures import ThreadPoolExecutor
        import os
        from pathlib import Path

        import pandas as pd
        from tqdm import tqdm

        from autrainer.datasets.utils import AbstractFileHandler
        from autrainer.transforms import SmartCompose

        def _get_csvs(path: str) -> List[str]:
            return [
                f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and f.endswith(".csv")
            ]

        def _read_csvs(path: str, files: List[str]) -> List[pd.DataFrame]:
            return [pd.read_csv(os.path.join(path, f)) for f in files]

        def _get_unique_files(
            dfs: List[pd.DataFrame],
            column: str,
        ) -> List[str]:
            return pd.concat(dfs, ignore_index=True)[column].unique().tolist()

        def _create_subdirs(path: str, files: List[str]) -> None:
            for f in files:
                os.makedirs(
                    os.path.dirname(os.path.join(path, f)),
                    exist_ok=True,
                )

        def _get_num_chunks() -> int:
            if self.num_workers == -1:
                return os.cpu_count() or 1
            return self.num_workers

        def _split_chunks(files: List[str], chunks: int) -> List[List[str]]:
            avg = len(files) / chunks
            out = []
            last = 0
            while last < len(files):
                out.append(files[int(last) : int(last + avg)])
                last += avg
            return out

        def _process_chunk(
            chunk: List[str],
            dataset: dict,
            preprocess: dict,
            pbar: tqdm,
            lock: Any,
        ) -> None:
            file_handler = autrainer.instantiate_shorthand(
                config=preprocess["file_handler"],
                instance_of=AbstractFileHandler,
            )
            output_file_handler = autrainer.instantiate_shorthand(
                config=dataset["file_handler"],
                instance_of=AbstractFileHandler,
            )
            pipeline = SmartCompose(
                [
                    autrainer.instantiate_shorthand(t)
                    for t in preprocess["pipeline"]
                ]
            )

            file_count = 0
            for file_path in chunk:
                in_path = Path(dataset["path"], "default", file_path)
                out_path = Path(
                    dataset["path"],
                    dataset["features_subdir"],
                    file_path,
                ).with_suffix("." + dataset["file_type"])
                data = file_handler.load(in_path)
                data = pipeline(data, 0)
                output_file_handler.save(out_path, data)
                file_count += 1
                if file_count % self.pbar_frequency == 0:
                    with lock:
                        pbar.update(self.pbar_frequency)
            with lock:
                pbar.update(file_count % self.pbar_frequency)

        print("Preprocessing datasets...")
        for (name, dataset), preprocess in zip(
            self.datasets.items(), self.preprocessing.values()
        ):
            print(f" - {name}")
            if preprocess is None or os.path.isdir(
                os.path.join(dataset["path"], dataset["features_subdir"])
            ):
                continue

            csvs = _get_csvs(dataset["path"])
            dfs = _read_csvs(dataset["path"], csvs)
            unique_files = _get_unique_files(dfs, dataset["index_column"])
            _create_subdirs(
                os.path.join(dataset["path"], dataset["features_subdir"]),
                unique_files,
            )
            num_chunks = _get_num_chunks()
            chunks = _split_chunks(unique_files, num_chunks)
            lock = tqdm.get_lock()
            with tqdm(
                total=len(unique_files), desc=name, disable=self.silent
            ) as pbar:
                with ThreadPoolExecutor(max_workers=num_chunks) as executor:
                    futures = [
                        executor.submit(
                            _process_chunk,
                            chunk,
                            dataset,
                            preprocess,
                            pbar,
                            lock,
                        )
                        for chunk in chunks
                    ]
                    for future in futures:
                        future.result()


@catch_cli_errors
def preprocess(
    override_kwargs: Optional[dict] = None,
    num_workers: int = -1,
    pbar_frequency: int = 100,
    silent: bool = False,
    cfg_launcher: bool = False,
    config_name: str = "config",
    config_path: Optional[str] = None,
) -> None:
    """Launch a data preprocessing configuration.

    Args:
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        num_workers: Number of workers to use for preprocessing.
            Defaults to -1.
        pbar_frequency: Frequency of progress bar updates. Defaults to 100.
        silent: Disable progress bar output. Defaults to False.
        cfg_launcher: Use the launcher specified in the configuration instead
            of the Hydra basic launcher. Defaults to False.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "config".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    cmd = "preprocess"
    if cfg_launcher:
        cmd += " -l"
    if silent:
        cmd += " -s"
    cmd += f" -n {num_workers} -p {pbar_frequency}"
    run_autrainer_hydra_cmd(cmd, override_kwargs, config_name, config_path)

from dataclasses import dataclass
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
            dirs = {os.path.dirname(os.path.join(path, f)) for f in files}

            for d in dirs:
                os.makedirs(d, exist_ok=True)

        def _split_chunks(files: List[str], chunks: int) -> List[List[str]]:
            if chunks == 1:
                return [files]
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
                del data
                file_count += 1
                if file_count % self.update_frequency == 0:
                    with lock:
                        pbar.update(self.update_frequency)
            with lock:
                pbar.update(file_count % self.update_frequency)

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
            chunks = _split_chunks(unique_files, self.num_workers)
            lock = tqdm.get_lock()
            with tqdm(
                total=len(unique_files),
                desc=name,
                disable=self.update_frequency == 0,
            ) as pbar:
                with ThreadPoolExecutor(self.num_workers) as executor:
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

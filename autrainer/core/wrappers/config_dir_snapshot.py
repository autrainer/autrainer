import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Optional

from hydra.experimental.callback import Callback
from omegaconf import DictConfig


def pop_config_dir_from_argv() -> Optional[str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", "-cd")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args.config_dir


class ConfigDirSnapshot(Callback):
    _instance = None
    _temp_dir: Optional[Path] = None

    def __new__(cls) -> "ConfigDirSnapshot":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._temp_dir = None
        return cls._instance

    def create_snapshot_dir(self) -> Optional[str]:
        """Create a snapshot of the config directory by copying all
        configuration files to a temporary directory.

        The snapshot is created by either copying the config directory
        specified by the `--config-dir` argument (and removing it from
        `sys.argv`) or by copying the `conf` directory in the current
        working directory if it exists.
        If no config directory is found, the method returns None.

        Returns:
            The path to the temporary directory containing the snapshot of
            the config directory, or None if no config directory was found.
        """
        if self._temp_dir and self._temp_dir.is_dir():
            return str(self._temp_dir)

        config_dir = pop_config_dir_from_argv()
        if config_dir is None and os.path.isdir("conf"):
            config_dir = "conf"
        if config_dir is None or not os.path.isdir(config_dir):
            return None  # nothing to snapshot

        config_dir = Path(config_dir).resolve()
        self._temp_dir = Path(tempfile.mkdtemp())

        for path in config_dir.rglob("*.yaml"):
            if not path.is_file():
                continue  # skip directories ending with .yaml
            relative_path = path.relative_to(config_dir)
            destination = self._temp_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        return str(self._temp_dir)

    def remove_snapshot_dir(self) -> None:
        """Remove the temporary config directory and all its contents.

        Automatically called at the end of the Hydra multirun.
        """
        try:
            if self._temp_dir and self._temp_dir.is_dir():
                shutil.rmtree(self._temp_dir)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Failed to remove temp config dir: {e}")  # noqa: G004, LOG015
        finally:
            self._temp_dir = None  # always try to reset the temp dir

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Called in MULTIRUN mode after all jobs returns.

        When using a launcher, this will be executed on local machine.
        """
        self.remove_snapshot_dir()

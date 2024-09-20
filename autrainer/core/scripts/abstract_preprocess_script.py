from dataclasses import dataclass
import shutil
import sys
import tempfile

from .abstract_script import AbstractScript


@dataclass
class PreprocessArgs:
    cfg_launcher: bool


class AbstractPreprocessScript(AbstractScript):
    def add_arguments(self) -> None:
        self.parser.add_argument(
            "-l",
            "--cfg-launcher",
            action="store_true",
            required=False,
            help=(
                "Use the launcher specified in the configuration instead of "
                "the Hydra basic launcher. Defaults to False."
            ),
        )

    def _override_launcher(self, args: PreprocessArgs) -> None:
        self._pre_sys_argv = sys.argv.copy()
        if not args.cfg_launcher:
            sys.argv = sys.argv + ["hydra/launcher=basic"]

        self.tempdir = tempfile.mkdtemp()
        sys.argv = sys.argv + [
            "hydra/hydra_logging=none",
            "hydra/job_logging=none",
            f"hydra.sweep.dir={self.tempdir}",
            f"hydra.sweep.subdir={self.tempdir}",
        ]

    def _clean_up(self) -> None:
        shutil.rmtree(self.tempdir)
        sys.argv = self._pre_sys_argv

    @staticmethod
    def _id_in_dict(d: dict, id: str) -> bool:
        for v in d.values():
            if v["id"] == id:
                return True
        return False

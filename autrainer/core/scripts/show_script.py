from dataclasses import dataclass
import os
import shutil

from omegaconf import OmegaConf

import autrainer
from autrainer.core.constants import NamingConstants

from .abstract_script import AbstractScript, MockParser
from .command_line_error import CommandLineError
from .utils import catch_cli_errors


@dataclass
class ShowArgs:
    directory: str
    config: str
    save: bool
    force: bool


class ShowScript(AbstractScript):
    def __init__(self) -> None:
        super().__init__(
            "show",
            "Show and save a global configuration.",
            epilog="Example: autrainer show model EfficientNet-B0 -s",
            dataclass=ShowArgs,
        )

    def add_arguments(self) -> None:
        self.parser.add_argument(
            "directory",
            type=str,
            help="The directory to list configurations from. Choose from:"
            + "\n - ".join([""] + sorted(NamingConstants().CONFIG_DIRS)),
        )
        self.parser.add_argument(
            "config",
            type=str,
            help=(
                "The global configuration to show. Configurations can be "
                "discovered using the 'autrainer list' command."
            ),
        )
        self.parser.add_argument(
            "-s",
            "--save",
            action="store_true",
            help="Save the global configuration to the local conf/ directory.",
        )
        self.parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help=(
                "Force overwrite local configuration if it exists "
                "in combination with -s/--save."
            ),
        )

    def main(self, args: ShowArgs) -> None:
        self._assert_config_directory(args.directory)
        path = self._get_path(args)
        self._assert_config_exists(path, args)
        self._show_configuration(path)
        self._save_configuration(path, args)

    def _get_path(self, args: ShowArgs) -> str:
        return os.path.join(
            os.path.dirname(autrainer.__path__[0]),
            "autrainer-configurations",
            args.directory,
            ShowScript._get_config_name(args.config),
        )

    @staticmethod
    def _get_config_name(config: str) -> str:
        if not config.endswith(".yaml"):
            return f"{config}.yaml"
        return config

    def _assert_config_exists(self, path: str, args: ShowArgs) -> None:
        if not os.path.exists(path):
            m = (
                f"No global configuration '{args.config}' found "
                f"for '{args.directory}'."
            )
            raise CommandLineError(self.parser, m, code=1)

    def _show_configuration(self, path: str) -> None:
        config = OmegaConf.load(path)
        print(OmegaConf.to_yaml(config))

    def _save_configuration(self, path: str, args: ShowArgs) -> None:
        if not args.save:
            return

        out_path = os.path.join(
            "conf",
            args.directory,
            ShowScript._get_config_name(args.config),
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path) and not args.force:
            m = (
                f"{args.directory} configuration '{args.config}' "
                "already exists. Use -f to overwrite."
            )
            raise CommandLineError(self.parser, m, code=1)
        shutil.copyfile(path, out_path)


@catch_cli_errors
def show(
    directory: str,
    config: str,
    save: bool = False,
    force: bool = False,
) -> None:
    """Show and save a global configuration.

    If called in a notebook, the function will not raise an error and print
    the error message instead.

    Args:
        directory: The directory to list configurations from. Choose from:
            :const:`~autrainer.core.constants.NamingConstants.CONFIG_DIRS`.
        config: The global configuration to show. Configurations can be
            discovered using the 'autrainer list' command.
        save: Save the global configuration to the local conf directory.
            Defaults to False.
        force: Force overwrite local configuration if it exists in combination
            with save=True. Defaults to False.

    Raises:
        CommandLineError: If the global configuration does not exist.
        CommandLineError: If while saving the local configuration, the
            configuration already exists and force is not set.
    """
    script = ShowScript()
    script.parser = MockParser()
    script.main(ShowArgs(directory, config, save, force))

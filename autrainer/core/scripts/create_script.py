from dataclasses import dataclass
import os
import shutil
from typing import List, Optional

import autrainer
from autrainer.core.constants import NamingConstants

from .abstract_script import AbstractScript, MockParser
from .command_line_error import CommandLineError
from .utils import catch_cli_errors


@dataclass
class CreateArgs:
    directories: List[str]
    empty: bool
    all: bool
    force: bool


class CreateScript(AbstractScript):
    def __init__(self) -> None:
        super().__init__(
            "create",
            "Create a new project with default configurations.",
            epilog="Example: autrainer create -e",
            dataclass=CreateArgs,
        )

    def add_arguments(self) -> None:
        self.parser.add_argument(
            "directories",
            type=str,
            nargs="*",
            help="Configuration directories to create. One or more of:"
            + "\n - ".join([""] + sorted(NamingConstants().CONFIG_DIRS)),
        )
        self.parser.add_argument(
            "-e",
            "--empty",
            action="store_true",
            help="Create an empty project without any configuration directory.",
        )
        self.parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            help="Create a project with all configuration directories.",
        )
        self.parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help=(
                "Force overwrite if the configuration directory "
                "already exists."
            ),
        )

    def main(self, args: CreateArgs) -> None:
        self._assert_valid_args(args)
        self._assert_config_directories(args)
        self._assert_mutually_exclusive(args)
        self._assert_directory_not_exists(args)
        self._create_directory(args)
        self._create_default_config()

    def _assert_valid_args(self, args: CreateArgs) -> None:
        if not args.directories and not args.empty and not args.all:
            raise CommandLineError(
                parser=self.parser,
                message="No configuration directories specified.",
            )

    def _assert_config_directories(self, args: CreateArgs) -> None:
        if args.empty or args.all:
            return
        for directory in args.directories:
            self._assert_config_directory(directory)

    def _assert_mutually_exclusive(self, args: CreateArgs) -> None:
        if args.empty and args.all:
            raise CommandLineError(
                parser=self.parser,
                message=(
                    "The flags -e/--empty and -a/--all "
                    "are mutually exclusive."
                ),
            )
        if args.directories and (args.empty or args.all):
            raise CommandLineError(
                parser=self.parser,
                message=(
                    "The flags -e/--empty and -a/--all "
                    "are mutually exclusive with configuration directories."
                ),
            )

    def _assert_directory_not_exists(self, args: CreateArgs) -> None:
        if args.force:
            return
        if os.path.exists("conf") and os.path.isdir("conf"):
            raise CommandLineError(
                parser=self.parser,
                message=(
                    "Directory 'conf' already exists. "
                    "Use -f/--force to overwrite it."
                ),
                code=1,
            )

    def _create_directory(self, args: CreateArgs) -> None:
        directories = []
        if args.empty:
            directories.append("conf")
        elif args.all:
            directories.extend(
                f"conf/{directory}"
                for directory in NamingConstants().CONFIG_DIRS
            )
        else:
            directories.extend(
                f"conf/{directory}" for directory in args.directories
            )

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _create_default_config(self) -> None:
        src = os.path.join(
            os.path.dirname(autrainer.__path__[0]),
            "autrainer-configurations",
            "config.yaml",
        )
        dst = os.path.join("conf", "config.yaml")
        shutil.copyfile(src, dst)


@catch_cli_errors
def create(
    directories: Optional[List[str]] = None,
    empty: bool = False,
    all: bool = False,
    force: bool = False,
) -> None:
    """Create a new project with default configurations.

    If called in a notebook, the function will not raise an error and print
    the error message instead.

    Args:
        directories: Configuration directories to create. One or more of:
            :const:`~autrainer.core.constants.NamingConstants.CONFIG_DIRS`.
            Defaults to None.
        empty: Create an empty project without any configuration directory.
            Defaults to False.
        all: Create a project with all configuration directories.
            Defaults to False.
        force: Force overwrite if the configuration directory already exists.
            Defaults to False.

    Raises:
        CommandLineError: If no configuration directories are specified and
            neither the empty nor all flags are set.
        CommandLineError: If the empty and all flags are set at the same time.
        CommandLineError: If the empty or all flags are set in combination with
            configuration directories.
        CommandLineError: If the configuration directory already exists and the
            force flag is not set.
    """

    script = CreateScript()
    script.parser = MockParser()
    script.main(CreateArgs(directories, empty, all, force))

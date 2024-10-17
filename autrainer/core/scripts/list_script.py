from dataclasses import dataclass
import fnmatch
import os

import autrainer
from autrainer.core.constants import NamingConstants

from .abstract_script import AbstractScript, MockParser
from .command_line_error import CommandLineError
from .utils import catch_cli_errors


@dataclass
class ListArgs:
    directory: str
    local_only: bool
    global_only: bool
    pattern: str


class ListScript(AbstractScript):
    def __init__(self) -> None:
        super().__init__(
            "list",
            "List local and global configurations.",
            epilog="Example: autrainer list model -g -p EfficientNet-B*",
            dataclass=ListArgs,
        )

    def add_arguments(self) -> None:
        self.parser.add_argument(
            "directory",
            type=str,
            help="The directory to list configurations from. Choose from:"
            + "\n - ".join([""] + sorted(NamingConstants().CONFIG_DIRS)),
        )
        self.parser.add_argument(
            "-l",
            "--local-only",
            action="store_true",
            help="List local configurations only.",
        )
        self.parser.add_argument(
            "-g",
            "--global-only",
            action="store_true",
            help="List global configurations only.",
        )
        self.parser.add_argument(
            "-p",
            "--pattern",
            type=str,
            default="*",
            required=False,
            metavar="P",
            help="Glob pattern to filter configurations.",
        )

    def main(self, args: ListArgs) -> None:
        self._assert_config_directory(args.directory)
        self._list_configs(args)

    def _list_configs(self, args: ListArgs) -> None:
        if args.local_only == args.global_only:
            self._list_global_configs(args)
            self._list_local_configs(args)
        elif args.global_only:
            self._list_global_configs(args)
        else:
            self._list_local_configs(args)

    def _list_global_configs(self, args: ListArgs) -> None:
        path = os.path.join(
            os.path.dirname(autrainer.__path__[0]),
            "autrainer-configurations",
            args.directory,
        )
        self._print_configs(args.directory, path, "global", args.pattern)

    def _list_local_configs(self, args: ListArgs) -> None:
        path = os.path.join("conf", args.directory)
        if os.path.exists(path):
            self._print_configs(
                args.directory,
                os.path.join("conf", args.directory),
                "local",
                args.pattern,
            )
            return

        if args.local_only and not args.global_only:
            m = (
                f"Local conf directory '{args.directory}' does not exist.\n"
                f" Use 'autrainer create {args.directory}' "
                "to create a new project."
            )
            raise CommandLineError(self.parser, message=m, code=1)

    def _print_configs(
        self,
        directory: str,
        path: str,
        loc: str,
        pattern: str,
    ) -> None:
        files = [
            f.replace(".yaml", "")
            for f in sorted(os.listdir(path))
            if f.endswith(".yaml")
        ]
        files = fnmatch.filter(files, pattern)
        files = sorted(files, key=lambda x: (x.lower(), len(x)))
        if not files:
            print(f"No {loc} '{directory}' configurations found.")
            return
        print(f"{loc.capitalize()} '{directory}' configurations:")
        for f in files:
            print(f" - {f}")


@catch_cli_errors
def list_configs(
    directory: str,
    local_only: bool = False,
    global_only: bool = False,
    pattern: str = "*",
) -> None:
    """List local and global configurations.

    If called in a notebook, the function will not raise an error and print
    the error message instead.

    Args:
        directory: The directory to list configurations from. Choose from:
            :const:`~autrainer.core.constants.NamingConstants.CONFIG_DIRS`.
        local_only: List local configurations only. Defaults to False.
        global_only: List global configurations only. Defaults to False.
        pattern: Glob pattern to filter configurations. Defaults to "*".

    Raises:
        CommandLineError: If the local configuration directory does not exist
            and local_only is True.
    """

    script = ListScript()
    script.parser = MockParser()
    script.main(ListArgs(directory, local_only, global_only, pattern))

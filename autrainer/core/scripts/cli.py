import argparse
from typing import List, Optional

import autrainer
from autrainer.core.scripts import (
    AbstractScript,
    CommandLineError,
    CreateScript,
    DeleteFailedScript,
    DeleteStatesScript,
    FetchScript,
    GroupScript,
    InferenceScript,
    ListScript,
    PostprocessScript,
    PreprocessScript,
    ShowScript,
    TrainScript,
    print_help_on_error,
)


class CLI:
    def __init__(self) -> None:
        self.scripts = self._init_scripts()
        self.parser = self._init_parser()

    def _init_scripts(self) -> List[AbstractScript]:
        return [
            # configuration management
            CreateScript(),
            ListScript(),
            ShowScript(),
            # preprocessing
            FetchScript(),
            PreprocessScript(),
            # training
            TrainScript(),
            # inference
            InferenceScript(),
            # postprocessing
            PostprocessScript(),
            DeleteFailedScript(),
            DeleteStatesScript(),
            GroupScript(),
        ]

    def _init_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="autrainer",
            description=(
                "A Modular and Extensible Deep Learning Toolkit "
                "for Computer Audition Tasks."
            ),
        )
        parser.error = print_help_on_error(parser)
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"%(prog)s {autrainer.__version__}",
        )
        subparsers = parser.add_subparsers(
            dest="command",
            metavar="<command>",
        )
        for c in self.scripts:
            p = subparsers.add_parser(
                c.command,
                help=c.description,
            )
            p = CLI._setup_subparser(p, c)
        return parser

    @staticmethod
    def _setup_subparser(
        parser: argparse.ArgumentParser,
        command: AbstractScript,
    ) -> None:
        if command.extended_description:
            parser.description = (
                command.description + "\n" + command.extended_description
            )
        else:
            parser.description = command.description
        parser.epilog = command.epilog
        parser.formatter_class = argparse.RawTextHelpFormatter
        parser.error = print_help_on_error(parser)
        command.init_parser(parser)
        command.add_arguments()

    def _get_command(self, name: str) -> Optional[AbstractScript]:
        return next((c for c in self.scripts if c.command == name), None)

    def main(self) -> None:
        args, unknown = self.parser.parse_known_args()
        args = vars(args)
        command = self._get_command(args.pop("command", None))
        if command:
            command.run(args, unknown)
        else:
            raise CommandLineError(self.parser, "No command provided.")


def main() -> None:
    cli = CLI()
    try:
        cli.main()
    except CommandLineError as e:
        e.handle()


def get_parser() -> argparse.ArgumentParser:
    cli = CLI()  # pragma: no cover
    return cli.parser  # pragma: no cover


if __name__ == "__main__":
    main()  # pragma: no cover

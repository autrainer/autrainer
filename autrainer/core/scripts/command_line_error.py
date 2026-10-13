import argparse
import sys
from types import MethodType
from typing import NoReturn


class CommandLineError(Exception):
    """Command line error.

    Args:
        parser: Argument parser.
        message: Error message.
        code: Exit code. Defaults to 2 in accordance with the UNIX standard to indicate
            incorrect usage.
    """

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        message: str = "",
        code: int = 2,
    ) -> None:
        super().__init__(message)
        self.parser = parser
        self.message = message
        self.help = help
        self.code = code

    def handle(self) -> NoReturn:
        """Handle the error and exit."""
        if self.code == 2:
            self.parser.print_help(sys.stderr)
            print(file=sys.stderr)
        if self.message:
            print(self.message, file=sys.stderr)
        sys.exit(self.code)


def print_help_on_error(parser: argparse.ArgumentParser) -> MethodType:
    """Print help on error.

    Args:
        parser: Argument parser.

    Returns:
        Closure that raises a CommandLineError with the given message.
    """

    def closure(self: argparse.ArgumentParser, message: str) -> NoReturn:
        if not message:
            raise CommandLineError(parser)

        message = message.capitalize()
        if not message.endswith("."):
            message += "."
        raise CommandLineError(parser, message)

    return MethodType(closure, parser)

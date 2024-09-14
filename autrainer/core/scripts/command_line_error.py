import argparse
from functools import wraps
import sys
from typing import Any, Callable, NoReturn, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


class CommandLineError(Exception):
    """Command line error.

    Args:
        parser (argparse.ArgumentParser): Argument parser.
        message (str): Error message.
        code (int, optional): Exit code. Defaults to 2 in accordance with the
        UNIX standard to indicate incorrect usage.
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


def print_help_on_error(
    parser: argparse.ArgumentParser,
) -> Callable[[str], NoReturn]:
    """Print help on error.

    Args:
        parser (argparse.ArgumentParser): Argument parser.

    Returns:
        Callable[[str], NoReturn]: Closure that raises a CommandLineError with
        the given message.
    """

    def closure(message: str = "") -> NoReturn:
        if not message:
            raise CommandLineError(parser)

        message = message.capitalize()
        if not message.endswith("."):
            message += "."
        raise CommandLineError(parser, message)

    return closure


def catch_cli_errors(func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except CommandLineError as e:
            print(e.message, file=sys.stderr)

    return wrapper

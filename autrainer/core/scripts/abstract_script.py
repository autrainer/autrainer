from abc import ABC, abstractmethod
import argparse
import sys
from typing import Any, TypeVar

from autrainer.core.constants import NamingConstants

from .command_line_error import CommandLineError


T = TypeVar("T")


class AbstractScript(ABC):
    """Abstract script class.

    Args:
        command: Command name.
        description: Command description.
        extended_description: Extended description.
        epilog: Command epilog. Defaults to "".
        dataclass: Dataclass type. Defaults to None.
        unknown_args: Allow to pass unknown arguments. Defaults to False.
    """

    def __init__(
        self,
        command: str,
        description: str,
        extended_description: str = "",
        epilog: str = "",
        dataclass: type[T] | None = None,
        unknown_args: bool = False,
    ) -> None:
        self.command = command
        self.description = description
        self.extended_description = extended_description
        self.epilog = epilog
        self.dataclass = dataclass
        self.unknown_args = unknown_args

    def init_parser(self, parser: argparse.ArgumentParser) -> None:
        """Initialize the parser.

        Args:
            parser: The parser.
        """
        self.parser = parser

    def add_arguments(self) -> None:  # noqa: B027
        """Add arguments to the parser."""

    def run(self, args: dict[str, Any], unknown: list[str]) -> None:
        """Run the script.

        Args:
            args: Parsed arguments.
            unknown: List of unknown arguments.
        """
        assert getattr(self, "parser", None) is not None
        if not self.unknown_args:
            self._assert_no_unknown_args(unknown)
        else:
            sys.argv = sys.argv[:1] + unknown

        self.main(
            self._create_dataclass(args, self.dataclass)
            if self.dataclass is not None
            else args
        )

    @abstractmethod
    def main(self, args: T) -> None:
        """Main function to run the script.

        Args:
            args: Parsed arguments as a dataclass instance or dict if no dataclass is
                provided.
        """

    def _create_dataclass(self, args: dict[str, Any], dataclass: type[T]) -> T:
        """Create a dataclass instance from parsed arguments.

        Args:
            args: Parsed arguments.
            dataclass: Dataclass type.

        Returns:
            Dataclass instance.
        """
        return dataclass(**args)

    def _assert_config_directory(self, directory: str) -> None:
        """Assert that the configuration directory is valid.

        Args:
            directory: Configuration directory.

        Raises:
            CommandLineError: If the configuration directory is invalid.
        """
        if directory not in NamingConstants().CONFIG_DIRS:
            raise CommandLineError(
                parser=self.parser,
                message=f"Invalid configuration directory '{directory}'.",
            )

    def _assert_no_unknown_args(self, unknown: list[str]) -> None:
        """Assert that there are no unknown arguments.

        Args:
            unknown: List of unknown arguments.

        Raises:
            CommandLineError: If there are unknown arguments.
        """
        if unknown:
            raise CommandLineError(
                parser=self.parser,
                message=f"Unknown arguments: {unknown}.",
            )

from abc import ABC, abstractmethod
import argparse
import sys
from typing import Any, Dict, List, Optional, Type, TypeVar

from autrainer.core.constants import NamingConstants

from .command_line_error import CommandLineError


T = TypeVar("T")


class AbstractScript(ABC):
    """Abstract script class.

    Args:
        command (str): Command name.
        description (str): Command description.
        extended_description (str, optional): Extended description.
        epilog (str, optional): Command epilog. Defaults to "".
        dataclass (Type[T], optional): Dataclass type. Defaults to None.
        unknown_args (bool, optional): Allow to pass unknown arguments.
        Defaults to False.
    """

    def __init__(
        self,
        command: str,
        description: str,
        extended_description: str = "",
        epilog: str = "",
        dataclass: Optional[Type[T]] = None,
        unknown_args: bool = False,
    ):
        self.command = command
        self.description = description
        self.extended_description = extended_description
        self.epilog = epilog
        self.dataclass = dataclass or dict
        self.unknown_args = unknown_args

    def init_parser(self, parser: argparse.ArgumentParser) -> None:
        """Initialize the parser.

        Args:
            parser (argparse.ArgumentParser): The parser.
        """
        self.parser = parser

    def add_arguments(self) -> None:
        """Add arguments to the parser."""

    def run(self, args: Dict[str, Any], unknown: List[str]) -> None:
        """Run the script.

        Args:
            args (Dict[str, Any]): Parsed arguments.
            unknown (List[str]): List of unknown arguments.
        """
        assert getattr(self, "parser", None) is not None
        if not self.unknown_args:
            self._assert_no_unknown_args(unknown)
        else:
            sys.argv = sys.argv[:1] + unknown
        if self.dataclass is not dict:
            args = self._create_dataclass(args, self.dataclass)
        self.main(args)

    @abstractmethod
    def main(self, args: T) -> None:
        """Main function to run the script.

        Args:
            args (T): Parsed arguments as a dataclass instance or dict if no
            dataclass is provided.
        """

    def _create_dataclass(self, args: Dict[str, Any], dataclass: Type[T]) -> T:
        """Create a dataclass instance from parsed arguments.

        Args:
            args (Dict[str, Any]): Parsed arguments.
            dataclass (T): Dataclass type.

        Returns:
            T: Dataclass instance.
        """
        return dataclass(**args)

    def _assert_config_directory(self, directory: str) -> None:
        """Assert that the configuration directory is valid.

        Args:
            directory (str): Configuration directory.

        Raises:
            CommandLineError: If the configuration directory is invalid.
        """
        if directory not in NamingConstants().CONFIG_DIRS:
            raise CommandLineError(
                parser=self.parser,
                message=f"Invalid configuration directory '{directory}'.",
            )

    def _assert_no_unknown_args(self, unknown: List[str]) -> None:
        """Assert that there are no unknown arguments.

        Args:
            unknown (List[str]): List of unknown arguments.

        Raises:
            CommandLineError: If there are unknown arguments.
        """
        if unknown:
            raise CommandLineError(
                parser=self.parser,
                message=f"Unknown arguments: {unknown}.",
            )


class MockParser:
    def add_argument(self, *args, **kwargs) -> None:
        pass

    def print_help(self, *args, **kwargs) -> None:
        pass

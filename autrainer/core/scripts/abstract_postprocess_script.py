from dataclasses import dataclass
import os
from typing import Generic, TypeVar

from .abstract_script import AbstractScript
from .command_line_error import CommandLineError


TArgs = TypeVar("TArgs", bound="AbstractPostprocessArgs")


@dataclass
class AbstractPostprocessArgs:
    results_dir: str
    experiment_id: str


class AbstractPostprocessScript(AbstractScript, Generic[TArgs]):
    def add_arguments(self) -> None:
        self.parser.add_argument(
            "results_dir",
            type=str,
            help="Path to grid search results directory.",
        )
        self.parser.add_argument(
            "experiment_id",
            type=str,
            help="ID of experiment to postprocess.",
        )

    def main(self, args: TArgs) -> None:
        self._assert_results_dir_exists(args)
        self._assert_experiment_id_exists(args)

    def _assert_results_dir_exists(self, args: AbstractPostprocessArgs) -> None:
        if os.path.exists(args.results_dir) and os.path.isdir(args.results_dir):
            return
        raise CommandLineError(
            self.parser,
            f"Results directory '{args.results_dir}' does not exist.",
            code=1,
        )

    def _assert_experiment_id_exists(self, args: AbstractPostprocessArgs) -> None:
        path = os.path.join(args.results_dir, args.experiment_id)
        if os.path.exists(path) and os.path.isdir(path):
            return
        raise CommandLineError(
            self.parser,
            (
                f"Experiment ID '{args.experiment_id}' does not exist "
                f"in the results directory '{args.results_dir}'."
            ),
            code=1,
        )

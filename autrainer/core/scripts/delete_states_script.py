from dataclasses import dataclass
import glob
import os
from typing import List, Optional, Union

from autrainer.core.scripts.abstract_script import MockParser

from .abstract_postprocess_script import (
    AbstractPostprocessArgs,
    AbstractPostprocessScript,
)
from .utils import catch_cli_errors


@dataclass
class DeleteStatesArgs(AbstractPostprocessArgs):
    keep_best: bool
    keep_runs: Optional[List[str]]
    keep_iterations: Optional[List[int]]


class DeleteStatesScript(AbstractPostprocessScript):
    def __init__(self) -> None:
        super().__init__(
            "rm-states",
            "Delete states (.pt files) from an experiment.",
            epilog="Example: autrainer rm-states results default",
            dataclass=DeleteStatesArgs,
        )

    def add_arguments(self) -> None:
        super().add_arguments()
        self.parser.add_argument(
            "-b",
            "--keep-best",
            action="store_false",
            default=True,
            help="Keep best states. Defaults to True.",
        )
        self.parser.add_argument(
            "-r",
            "--keep-runs",
            nargs="+",
            metavar="R",
            default=None,
            type=str,
            help="Runs to keep.",
        )
        self.parser.add_argument(
            "-i",
            "--keep-iterations",
            nargs="+",
            metavar="I",
            default=None,
            type=int,
            help="Iterations to keep.",
        )

    def main(self, args: DeleteStatesArgs) -> None:
        super().main(args)
        self._clean_states(args)

    def _clean_states(self, args: DeleteStatesArgs) -> None:
        state_paths = self._collect_state_paths(args)
        if not state_paths:
            return
        if args.keep_best:
            state_paths = self._filter_best_states(state_paths)
        if args.keep_runs:
            state_paths = self._filter_runs(state_paths, args.keep_runs)
        if args.keep_iterations:
            state_paths = self._filter_iterations(
                state_paths,
                args.keep_iterations,
            )
        if self._confirm_deletion(state_paths):
            self._delete_states(state_paths)

    def _collect_state_paths(self, args: DeleteStatesArgs) -> List[str]:
        pattern = "**/*.pt"
        path = os.path.join(args.results_dir, args.experiment_id, "training")
        return [
            f for f in glob.glob(os.path.join(path, pattern), recursive=True)
        ]

    def _filter_best_states(self, state_paths: List[str]) -> List[str]:
        return [f for f in state_paths if "_best" != self._get_iter_dir(f)]

    def _filter_runs(
        self,
        state_paths: List[str],
        runs: List[str],
    ) -> List[str]:
        for run in runs:
            state_paths = [
                f for f in state_paths if run != self._get_run_name(f)
            ]
        return state_paths

    def _filter_iterations(
        self,
        state_paths: List[str],
        iterations: List[int],
    ) -> List[str]:
        for iteration in iterations:
            state_paths = [
                f for f in state_paths if iteration != self._get_iter_num(f)
            ]
        return state_paths

    def _get_iter_dir(self, state_path: str) -> str:
        return os.path.basename(os.path.dirname(state_path))

    def _get_run_name(self, state_path: str) -> str:
        return os.path.basename(os.path.dirname(os.path.dirname(state_path)))

    def _get_iter_num(self, state_path: str) -> Union[int, str]:
        iter_dir = self._get_iter_dir(state_path)
        if iter_dir in ["_best", "_initial"]:
            return iter_dir
        return int(iter_dir.split("_")[-1])

    def _confirm_deletion(self, state_paths: List[str]) -> bool:
        print("The following states will be deleted:")
        print("\n".join(state_paths))
        print()
        response = input(
            "Are you sure you want to delete "
            f"{len(state_paths)} states? (y/n): "
        ).lower()
        return response in ["y", "yes"]

    def _delete_states(self, state_paths: List[str]) -> None:
        for state_path in state_paths:
            try:
                os.remove(state_path)
                print(f"Deleted: {state_path}")
            except Exception as e:
                print(f"Error deleting {state_path}: {e}")  # pragma: no cover


@catch_cli_errors
def rm_states(
    results_dir: str,
    experiment_id: str,
    keep_best: bool = True,
    keep_runs: Optional[List[str]] = None,
    keep_iterations: Optional[List[int]] = None,
) -> None:
    """Delete states (.pt files) from an experiment.

    If called in a notebook, the function will not raise an error and print
    the error message instead.

    Args:
        results_dir: Path to grid search results directory.
        experiment_id: ID of experiment to postprocess.
        keep_best: Keep best states. Defaults to True.
        keep_runs: Runs to keep. Defaults to None.
        keep_iterations: Iterations to keep. Defaults to None.

    Raises:
        CommandLineError: If the results directory or experiment ID dont exist.
    """
    script = DeleteStatesScript()
    script.parser = MockParser()
    script.main(
        DeleteStatesArgs(
            results_dir,
            experiment_id,
            keep_best,
            keep_runs,
            keep_iterations,
        )
    )

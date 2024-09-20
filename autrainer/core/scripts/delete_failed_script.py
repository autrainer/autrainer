from dataclasses import dataclass
import os
import shutil
from typing import List

from autrainer.core.scripts.abstract_script import MockParser

from .abstract_postprocess_script import (
    AbstractPostprocessArgs,
    AbstractPostprocessScript,
)
from .utils import catch_cli_errors


@dataclass
class DeleteFailedArgs(AbstractPostprocessArgs):
    force: bool


class DeleteFailedScript(AbstractPostprocessScript):
    def __init__(self) -> None:
        super().__init__(
            "rm-failed",
            "Delete failed runs from an experiment.",
            epilog="Example: autrainer rm-failed results default",
            dataclass=DeleteFailedArgs,
        )

    def add_arguments(self) -> None:
        super().add_arguments()
        self.parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            default=False,
            help=(
                "Force deletion of failed runs without confirmation. "
                "Defaults to False."
            ),
        )

    def main(self, args: DeleteFailedArgs) -> None:
        super().main(args)
        self._clean_failed(args)

    def _clean_failed(self, args: DeleteFailedArgs) -> None:
        failed_runs = self._collect_failed_runs(args)
        if not failed_runs:
            return
        if args.force or self._confirm_deletion(failed_runs):
            self._delete_runs(args, failed_runs)

    def _collect_failed_runs(self, args: DeleteFailedArgs) -> List[str]:
        path = os.path.join(args.results_dir, args.experiment_id, "training")
        return [
            f
            for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
            and not os.path.exists(os.path.join(path, f, "metrics.csv"))
        ]

    def _confirm_deletion(self, failed_runs: List[str]) -> bool:
        print("The following failed runs will be deleted:")
        print("\n".join(failed_runs))
        print()
        response = input(
            "Are you sure you want to delete "
            f"{len(failed_runs)} failed runs? (y/n): "
        ).lower()
        return response in ["y", "yes"]

    def _delete_runs(
        self,
        args: DeleteFailedArgs,
        failed_runs: List[str],
    ) -> None:
        path = os.path.join(args.results_dir, args.experiment_id, "training")
        for run in failed_runs:
            shutil.rmtree(os.path.join(path, run))


@catch_cli_errors
def rm_failed(
    results_dir: str,
    experiment_id: str,
    force: bool = False,
) -> None:
    """Delete failed runs from an experiment.

    If called in a notebook, the function will not raise an error and print
    the error message instead.

    Args:
        results_dir: Path to grid search results directory.
        experiment_id: ID of experiment to postprocess.
        force: Force deletion of failed runs without confirmation.
            Defaults to False.

    Raises:
        CommandLineError: If the results directory or experiment ID dont exist.
    """
    script = DeleteFailedScript()
    script.parser = MockParser()
    script.main(
        DeleteFailedArgs(
            results_dir,
            experiment_id,
            force,
        )
    )

from dataclasses import dataclass
from typing import List, Optional

from autrainer.core.constants import NamingConstants
from autrainer.core.scripts.abstract_script import MockParser

from .abstract_postprocess_script import (
    AbstractPostprocessArgs,
    AbstractPostprocessScript,
)
from .command_line_error import CommandLineError
from .utils import catch_cli_errors


@dataclass
class PostprocessArgs(AbstractPostprocessArgs):
    max_runs: Optional[int]
    aggregate: Optional[List[List[str]]]


class PostprocessScript(AbstractPostprocessScript):
    def __init__(self) -> None:
        super().__init__(
            "postprocess",
            "Postprocess grid search results.",
            epilog="Example: autrainer postprocess results default -a seed",
            dataclass=PostprocessArgs,
        )

    def add_arguments(self) -> None:
        super().add_arguments()
        self.parser.add_argument(
            "-m",
            "--max-runs",
            type=int,
            metavar="N",
            required=False,
            help="Maximum number of best runs to plot.",
            default=None,
        )
        self.parser.add_argument(
            "-a",
            "--aggregate",
            type=str,
            metavar="A",
            required=False,
            nargs="+",
            action="append",
            help="Configurations to aggregate. One or more of:"
            + "\n - ".join(
                [""] + sorted(NamingConstants().VALID_AGGREGATIONS)
            ),
        )

    def main(self, args: PostprocessArgs) -> None:
        super().main(args)
        self._assert_valid_aggregations(args)
        self._summarize(args)

    def _assert_valid_aggregations(self, args: PostprocessArgs) -> None:
        if not args.aggregate:
            return
        for agg in args.aggregate:
            for a in agg:
                if a not in NamingConstants().VALID_AGGREGATIONS:
                    m = f"Invalid configuration to aggregate: '{a}'."
                    raise CommandLineError(self.parser, m)

    def _summarize(self, args: PostprocessArgs) -> None:
        from autrainer.postprocessing import (
            AggregateGrid,
            SummarizeGrid,
        )

        sg = SummarizeGrid(
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
            max_runs_plot=args.max_runs,
        )
        sg.summarize()
        sg.plot_aggregated_bars()
        sg.plot_metrics()

        if not args.aggregate:
            return
        for agg in args.aggregate:
            ag = AggregateGrid(
                results_dir=args.results_dir,
                experiment_id=args.experiment_id,
                aggregate_list=agg,
                max_runs_plot=args.max_runs,
            )
            ag.aggregate()
            ag.summarize()


@catch_cli_errors
def postprocess(
    results_dir: str,
    experiment_id: str,
    max_runs: Optional[int] = None,
    aggregate: Optional[List[List[str]]] = None,
) -> None:
    """Postprocess grid search results.

    If called in a notebook, the function will not raise an error and print
    the error message instead.

    Args:
        results_dir: Path to grid search results directory.
        experiment_id: ID of experiment to postprocess.
        max_runs: Maximum number of best runs to plot. Defaults to None.
        aggregate: Configurations to aggregate. One or more of:
            :const:`~autrainer.core.constants.NamingConstants.VALID_AGGREGATIONS`.
            Defaults to None.

    Raises:
        CommandLineError: If the results directory or experiment ID dont exist.
    """
    script = PostprocessScript()
    script.parser = MockParser()
    script.main(
        PostprocessArgs(
            results_dir,
            experiment_id,
            max_runs,
            aggregate,
        )
    )

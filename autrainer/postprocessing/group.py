import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, ListConfig

from .aggregate import AggregateGrid
from .postprocessing_utils import load_yaml
from .summarize import SummarizeGrid


class GroupGrid:
    def __init__(
        self,
        results_dir: str,
        groupings: Union[ListConfig[DictConfig], List[Dict]],
        max_runs: Optional[int] = None,
        plot_params: Optional[dict] = None,
    ) -> None:
        """Group runs of one or more grid search experiments based on the
        specified groupings.

        Args:
            results_dir: The directory where the results are stored.
            groupings: A list of experiments to create containing one or more
                runs to group.
            max_runs_plot: The maximum number of best runs to plot. If None,
                all runs will be plotted. Defaults to None.
            plot_params: Additional parameters for plotting. Defaults to None.
        """
        self.results_dir = results_dir
        self.groupings = groupings
        self.max_runs = max_runs
        self.plot_params = plot_params

    def group_runs(self) -> None:
        """Group the runs of the specified experiments based on the groupings."""
        for grouping in self.groupings:
            grouping = self._preprocess(grouping)
            mappings = self._copy_runs(grouping.experiment_id, grouping.runs)
            self._group_experiment(
                grouping.experiment_id,
                grouping.runs,
                grouping.create_summary,
                mappings,
            )

    def _preprocess(self, grouping: DictConfig) -> DictConfig:
        set_dir = grouping.get("dir") is not None
        set_id = grouping.get("id") is not None
        set_states = grouping.get("states") is not None

        if not set_dir and not set_id and not set_states:
            return grouping

        for run in grouping.runs:
            if set_dir and run.get("dir") is None:
                run.dir = grouping.dir
            if set_id and run.get("id") is None:
                run.id = grouping.id
            if set_states and run.get("states") is None:
                run.states = grouping.states
        return grouping

    def _copy_runs(
        self, experiment: str, runs: ListConfig[DictConfig]
    ) -> Dict[str, List[str]]:
        run_paths, copy_states = self._resolve_runs_paths(experiment, runs)
        run_paths, copy_states, mappings = self._resolve_aggregated_runs(
            run_paths, copy_states
        )
        for run_path, copy_state in zip(run_paths, copy_states):
            self._copy_run(experiment, run_path, copy_state)
        return mappings

    def _resolve_runs_paths(
        self, experiment: str, runs: ListConfig[DictConfig]
    ) -> Tuple[List[str], List[bool]]:
        paths = []
        copy_states = []
        for run in runs:
            base_path = os.path.join(run.dir, run.id)
            dirs = [d for d in os.listdir(base_path) if d.startswith("agg_")]
            dirs.append("training")
            if isinstance(run.combine, str):
                run.combine = [run.combine]
            for r in run.combine:
                for d in dirs:
                    p = os.path.join(base_path, d, r)
                    if os.path.exists(os.path.join(p, "metrics.csv")):
                        paths.append(p)
                        copy_states.append(run.states)
                        break
                else:
                    raise ValueError(
                        f"Run '{r}' to combine to '{run.run_name}'"
                        f"in experiment '{experiment}' does not exist "
                        f"in experiment '{run.id}'"
                    )
        return paths, copy_states

    def _resolve_aggregated_runs(
        self, run_paths: List[str], copy_states: List[bool]
    ) -> Tuple[List[str], List[bool], Dict[str, List[str]]]:
        expanded_paths = []
        expanded_states = []
        mappings = {}
        for run_path, copy_state in zip(run_paths, copy_states):
            mappings[os.path.basename(run_path)] = []
            for run in self._get_aggregated_runs(run_path):
                path = os.path.join(
                    os.path.dirname(os.path.dirname(run_path)), "training", run
                )
                expanded_paths.append(path)
                expanded_states.append(copy_state)
                mappings[os.path.basename(run_path)].append(run)

        return expanded_paths, expanded_states, mappings

    def _get_aggregated_runs(self, run_path: str) -> List[str]:
        if "#" not in os.path.basename(run_path):
            return [os.path.basename(run_path)]
        return load_yaml(os.path.join(run_path, "runs.yaml"))

    def _copy_run(
        self, experiment: str, run_path: str, copy_state: bool
    ) -> None:
        dst = os.path.join(
            self.results_dir,
            experiment,
            "training",
            os.path.basename(run_path),
        )
        if os.path.exists(
            os.path.join(dst, "metrics.csv")
        ) or not os.path.exists(run_path):
            return
        ignore = None if copy_state else shutil.ignore_patterns("*.pt")
        shutil.copytree(
            src=run_path,
            dst=dst,
            ignore=ignore,
        )

    def _group_experiment(
        self,
        experiment: str,
        runs: ListConfig[DictConfig],
        create_summary: bool,
        mappings: Dict[str, List[str]],
    ) -> None:
        grouped_dict = {}
        for run in runs:
            base_runs = []
            for r in run.combine:
                base_runs.extend(mappings[r])

            grouped_dict.update({run.run_name: base_runs})
        if create_summary:
            sg = SummarizeGrid(
                results_dir=self.results_dir,
                experiment_id=experiment,
                max_runs_plot=self.max_runs,
            )
            sg.summarize()
            sg.plot_aggregated_bars()
            sg.plot_metrics()

        ag = AggregateGrid(
            results_dir=self.results_dir,
            experiment_id=experiment,
            aggregate_list="",
            aggregate_name="grouping",
            aggregated_dict=grouped_dict,
            max_runs_plot=self.max_runs,
            plot_params=self.plot_params,
        )
        ag.aggregate()
        ag.summarize()

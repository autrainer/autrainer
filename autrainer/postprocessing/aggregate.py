from collections import defaultdict
import os
import shutil
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf
import pandas as pd

import autrainer
from autrainer.core.constants import NamingConstants
from autrainer.core.plotting import PlotMetrics
from autrainer.core.utils import Timer
from autrainer.loggers import AbstractLogger
from autrainer.metrics import AbstractMetric

from .postprocessing_utils import (
    get_plotting_params,
    get_run_names,
    get_training_type,
    load_yaml,
    save_yaml,
)
from .summarize import SummarizeGrid


class AggregateGrid:
    def __init__(
        self,
        results_dir: str,
        experiment_id: str,
        aggregate_list: List[str],
        aggregate_prefix: str = "agg",
        training_dir: str = "training",
        max_runs_plot: Optional[int] = None,
        aggregate_name: Optional[str] = None,
        aggregated_dict: Optional[dict] = None,
        plot_params: Optional[dict] = None,
    ) -> None:
        """Aggregate the results of a grid search over one or more parameters.

        If loggers have been used for the grid search, the aggregated results
        will be logged to the same loggers.

        Args:
            results_dir: The directory where the results are stored.
            experiment_id: The ID of the grid search experiment.
            aggregate_list: The list of parameters to aggregate over.
            aggregate_prefix: The prefix for the aggregated experiment ID.
                Defaults to "agg".
            training_dir: The directory of the training results of the
                experiment. Defaults to "training".
            max_runs_plot: The maximum number of best runs to plot. If None,
                all runs will be plotted. Defaults to None.
            aggregate_name: The name of the aggregated experiment. If None,
                it will be generated from the aggregate_list. Defaults to None.
            aggregated_dict: A dictionary mapping the aggregated experiment
                names to the runs to aggregate. If None, the runs will be
                aggregated based on the aggregate_list. Defaults to None.
            plot_params: Additional parameters for plotting. Defaults to None.
        """
        self.aggregate_list = aggregate_list
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.aggregate_name = aggregate_name or "_".join(
            (aggregate_prefix, *self.aggregate_list)
        )
        self.aggregated_dict = aggregated_dict
        self.plot_params = plot_params
        self.output_directory = os.path.join(
            self.results_dir, self.experiment_id, self.aggregate_name
        )
        self.training_directory = os.path.join(
            self.results_dir, self.experiment_id, training_dir
        )
        self.max_runs_plot = max_runs_plot
        if os.path.exists(self.output_directory):
            shutil.rmtree(self.output_directory)
        os.makedirs(self.output_directory, exist_ok=True)
        self.run_names = get_run_names(self.training_directory)
        self.training_type = get_training_type(
            self.training_directory, self.run_names
        )
        self.run_names.sort()
        if self.plot_params is None:
            self.plot_params = get_plotting_params(
                self.training_directory, self.run_names[0]
            )

    def aggregate(self) -> None:
        """Aggregate the runs based on the specified parameters."""
        if self.aggregated_dict is not None:
            aggregated_runs = self.aggregated_dict
        else:
            aggregated_runs = self._aggregate_run_names(self.aggregate_list)
        for agg_name, run_list in aggregated_runs.items():
            self._aggregate_best(agg_name, run_list)
            self._aggregate_test(agg_name, run_list)
            self._aggregate_config(agg_name, run_list)
            self._aggregate_timer(agg_name, run_list)
            self._aggregate_metrics(agg_name, run_list)
            save_yaml(
                os.path.join(self.output_directory, agg_name, "runs.yaml"),
                run_list,
            )

    def summarize(self) -> None:
        """Summarize the aggregated results."""
        sg = SummarizeGrid(
            results_dir=self.results_dir,
            experiment_id=self.experiment_id,
            training_dir=self.aggregate_name,
            summary_dir=self.aggregate_name,
            clear_old_outputs=False,
            training_type=self.training_type,
            max_runs_plot=self.max_runs_plot,
            plot_params=self.plot_params,
        )
        sg.summarize()
        sg.plot_metrics()

    def _check_if_valid_aggregation(self, over: list) -> None:
        for o in over:
            if o not in NamingConstants().VALID_AGGREGATIONS:
                raise ValueError(f"Invalid aggregation: {o}")

    def _aggregate_run_names(self, over: list) -> dict:
        self._check_if_valid_aggregation(over)
        param_dict = {
            p: i for i, p in enumerate(NamingConstants().NAMING_CONVENTION)
        }
        over_idxs = [param_dict[p] for p in over]
        aggregated = defaultdict(list)
        for run_name in self.run_names:
            params = run_name.split("_")
            for idx in over_idxs:
                params[idx] = "#"
            agg_key = "_".join(params)
            aggregated[agg_key].append(run_name)
        return aggregated

    def _aggregate_best(self, agg_name: str, run_list: list):
        os.makedirs(
            os.path.join(self.output_directory, agg_name, "_best"),
            exist_ok=True,
        )
        metrics = self._aggregate_yaml(run_list, "_best/dev.yaml", "dev")
        save_yaml(
            os.path.join(self.output_directory, agg_name, "_best", "dev.yaml"),
            metrics,
        )

    def _aggregate_test(self, agg_name: str, run_list: list):
        path = os.path.join(self.output_directory, agg_name, "_test")
        os.makedirs(path, exist_ok=True)
        metrics = self._aggregate_yaml(
            run_list, "_test/test_holistic.yaml", "test"
        )
        save_yaml(os.path.join(path, "test_holistic.yaml"), metrics)

    def _aggregate_yaml(self, run_list: list, path: str, yaml_type: str):
        assert yaml_type in ["dev", "test"]
        loss_type = "dev_loss" if yaml_type == "dev" else "loss"
        dfs = []
        for run in run_list:
            metrics = load_yaml(
                os.path.join(self.training_directory, run, path)
            )
            dfs.append(pd.DataFrame(metrics))
        df = pd.concat(dfs, keys=run_list, names=["run", "type"])
        df_mean = df.groupby(level="type").mean().reset_index()
        df_std = df.groupby(level="type").std().fillna(0).reset_index()
        df_std["type"] = df_std["type"].apply(lambda x: f"{x}.std")
        df = pd.concat([df_mean, df_std]).set_index(["type"])
        metrics = df.to_dict()
        metrics[loss_type] = {
            k: v for k, v in metrics[loss_type].items() if "all" == k
        }
        if yaml_type == "dev":
            metrics["iteration"] = {
                k: v for k, v in metrics["iteration"].items() if "all" == k
            }
        return metrics

    def _aggregate_config(self, agg_name: str, run_list: list):
        path = os.path.join(self.output_directory, agg_name, ".hydra")
        os.makedirs(path, exist_ok=True)
        runs = [
            OmegaConf.load(
                os.path.join(
                    self.training_directory, r, ".hydra", "config.yaml"
                )
            )
            for r in run_list
        ]
        config = runs.pop(0)
        if self.aggregated_dict is None:
            for key in self.aggregate_list:
                config[key] = self._replace_differing_values(config, runs, key)
        save_yaml(
            os.path.join(path, "config.yaml"), OmegaConf.to_container(config)
        )

    def _replace_differing_values(
        self,
        config: DictConfig,
        runs: List[DictConfig],
        key: str,
    ) -> DictConfig:
        if not runs:
            return config[key]

        if not isinstance(config[key], DictConfig):
            return "#"

        for k in config[key].keys():
            if not isinstance(config[key][k], DictConfig):
                values = [r[key][k] for r in runs + [config]]
                if len(set(values)) > 1:
                    config[key][k] = "#"
            else:
                config[key][k] = self._replace_differing_values(
                    config[key], [r[key] for r in runs], k
                )

        return config[key]

    def _aggregate_timer(self, agg_name: str, run_list: list):
        mean_timer = {
            "train": {"mean_seconds": 0, "total_seconds": 0},
            "dev": {"mean_seconds": 0, "total_seconds": 0},
            "test": {"mean_seconds": 0, "total_seconds": 0},
        }
        for run in run_list:
            timers = load_yaml(
                os.path.join(self.training_directory, run, "timer.yaml")
            )
            for k, v in timers.items():
                mean_timer[k]["mean_seconds"] += v["mean_seconds"]
                mean_timer[k]["total_seconds"] += v["total_seconds"]
        for k, v in mean_timer.items():
            v["mean_seconds"] /= len(run_list)
            v["mean"] = Timer.pretty_time(v["mean_seconds"])
            v["total_seconds"] /= len(run_list)
            v["total"] = Timer.pretty_time(v["total_seconds"])
        save_yaml(
            os.path.join(self.output_directory, agg_name, "timer.yaml"),
            mean_timer,
        )

    def _aggregate_metrics(self, agg_name: str, run_list: list):
        dfs = []
        for run in run_list:
            df = pd.read_csv(
                os.path.join(self.training_directory, run, "metrics.csv"),
                index_col="iteration",
            )
            dfs.append(df)
        cfg = OmegaConf.load(
            os.path.join(
                self.output_directory, agg_name, ".hydra", "config.yaml"
            )
        )
        if isinstance(cfg.dataset.metrics, str):
            raise ValueError(
                "Unable to aggregate over datasets with different metrics."
            )
        metric_fns = [
            autrainer.instantiate_shorthand(m, instance_of=AbstractMetric)
            for m in cfg.dataset.metrics
        ]
        tracking_metric = autrainer.instantiate_shorthand(
            config=cfg.dataset.tracking_metric,
            instance_of=AbstractMetric,
        )

        df = pd.concat(dfs, keys=run_list, names=["run", "iteration"])
        mean_df = df.groupby(level="iteration").mean()
        std_df = df.groupby(level="iteration").std().fillna(0)
        df = mean_df.join(std_df, rsuffix=".std")
        df["iteration"] = df.index
        df.to_csv(
            os.path.join(self.output_directory, agg_name, "metrics.csv"),
            index=False,
        )
        df.drop(columns=["iteration"], inplace=True)

        plotter = PlotMetrics(
            output_directory=os.path.join(self.output_directory, agg_name),
            training_type=self.training_type,
            metric_fns=metric_fns,
            **self.plot_params,
        )
        plotter.plot_run(df)

        loggers = []
        for logger in cfg.get("loggers", []):
            loggers.append(
                autrainer.instantiate_shorthand(
                    logger,
                    instance_of=AbstractLogger,
                    exp_name=self.experiment_id + "." + self.aggregate_name,
                    run_name=agg_name,
                    metrics=metric_fns,
                    tracking_metric=tracking_metric,
                )
            )
        timers = load_yaml(
            os.path.join(self.output_directory, agg_name, "timer.yaml")
        )
        test_metrics = load_yaml(
            os.path.join(
                self.output_directory, agg_name, "_test", "test_holistic.yaml"
            )
        )
        for logger in loggers:
            logger.setup()
            logger.log_params(cfg)
            logger.log_timers(
                {
                    "time.train.mean": timers["train"]["mean"],
                    "time.dev.mean": timers["dev"]["mean"],
                    "time.test.mean": timers["test"]["mean"],
                }
            )
            for iteration in df.index:
                metrics = df.loc[iteration].to_dict()
                metrics = {
                    k: v for k, v in metrics.items() if not k.endswith(".std")
                }
                logger.log_metrics(metrics, iteration)
            logger.log_metrics(
                {"test_" + k: v["all"] for k, v in test_metrics.items()}
            )
            logger.log_artifact(
                os.path.join(agg_name, ".hydra", "config.yaml"),
                self.output_directory,
            )
            logger.log_artifact(
                os.path.join(agg_name, "metrics.csv"), self.output_directory
            )
            logger.end_run()

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from autrainer.metrics import AbstractMetric

from .plot_base import PlotBase


class PlotMetrics(PlotBase):
    def __init__(
        self,
        output_directory: str,
        training_type: str,
        figsize: tuple,
        latex: bool,
        filetypes: list,
        pickle: bool,
        context: str,
        palette: str,
        replace_none: bool,
        add_titles: bool,
        add_xlabels: bool,
        add_ylabels: bool,
        rcParams: dict,
        metric_fns: List[AbstractMetric],
    ) -> None:
        """Plot the metrics of one or multiple runs.

        Args:
            output_directory: Output directory to save plots to.
            training_type: Type of training in ["Epoch", "Step"].
            figsize: Figure size in inches.
            latex: Whether to use LaTeX in plots. Requires the `latex` package.
                To install all necessary dependencies, run:
                `pip install autrainer[latex]`.
            filetypes: Filetypes to save plots as.
            pickle: Whether to save additional pickle files of the plots.
            context: Context for seaborn plots.
            palette: Color palette for seaborn plots.
            replace_none: Whether to replace "None" in labels with "~".
            add_titles: Whether to add titles to plots.
            add_xlabels: Whether to add x-labels to plots.
            add_ylabels: Whether to add y-labels to plots.
            rcParams: Additional Matplotlib rcParams to set.
            metric_fns: List of metrics to use for plotting.
        """
        super().__init__(
            output_directory,
            training_type,
            figsize,
            latex,
            filetypes,
            pickle,
            context,
            palette,
            replace_none,
            add_titles,
            add_xlabels,
            add_ylabels,
            rcParams,
        )
        self.metric_fns = metric_fns

    def plot_run(self, metrics: pd.DataFrame, std_scale: float = 0.1) -> None:
        """Plot the metrics of a single run.

        Args:
            metrics: DataFrame containing the metrics.
            std_scale: Scale factor for the standard deviation.
                Defaults to 0.1.
        """
        std_columns = metrics.filter(regex="\\.std$").columns
        has_std = not std_columns.empty

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        sns.lineplot(
            data=metrics[["train_loss", "dev_loss"]], ax=ax, dashes=False
        )
        if has_std:
            for col in ["train_loss", "dev_loss"]:
                ax.fill_between(
                    metrics.index,
                    metrics[col] - std_scale * metrics[f"{col}.std"],
                    metrics[col] + std_scale * metrics[f"{col}.std"],
                    alpha=0.2,
                )
        self._add_label(ax, self.training_type, "loss")
        self.save_plot(fig, "loss", path="_plots")

        for key in metrics.columns:
            if ".std" in key or key in ["train_loss", "dev_loss", "iteration"]:
                continue
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            sns.lineplot(data=metrics[key], ax=ax, dashes=False)
            if has_std:
                ax.fill_between(
                    metrics.index,
                    metrics[key] - std_scale * metrics[f"{key}.std"],
                    metrics[key] + std_scale * metrics[f"{key}.std"],
                    alpha=0.2,
                )
            self._add_label(ax, self.training_type, key)
            self.save_plot(fig, key, path="_plots")

    def plot_metric(
        self,
        metrics: pd.DataFrame,
        metric: str,
        metrics_std: Optional[pd.DataFrame] = None,
        std_scale: float = 0.1,
        max_runs: Optional[int] = None,
    ) -> None:
        """Plot a single metric of multiple runs.

        Args:
            metrics: DataFrame containing the metrics.
            metric: Metric to plot.
            metrics_std: DataFrame containing the standard deviations.
                Defaults to None.
            std_scale: Scale factor for the standard deviation.
                Defaults to 0.1.
            max_runs: Maximum number of best runs to plot. If None, all runs
                are plotted. Defaults to None.
        """
        fig = plt.figure(figsize=self.figsize)

        if max_runs is None:
            max_runs = len(metrics.columns)
        metrics, metrics_std = self._select_top_runs(
            metric, metrics, metrics_std, max_runs
        )

        sns.lineplot(data=metrics, dashes=False)

        if metrics_std is not None:
            for col in metrics_std.columns:
                plt.fill_between(
                    metrics.index,
                    metrics[col] - std_scale * metrics_std[col],
                    metrics[col] + std_scale * metrics_std[col],
                    alpha=0.2,
                )

        self._add_label(plt.gca(), self.training_type, metric)
        path = os.path.join("plots", "training_plots")
        self.save_plot(fig, metric, path)

    def plot_aggregated_bars(
        self,
        metrics_df: pd.DataFrame,
        metric: str,
        subplots_by: int = 0,
        group_by: int = 1,
        split_subgroups: bool = True,
    ):
        """Plot aggregated bar plots for a metric.

        Generate a bar plots from the metrics_df, which are divided
        by the "subplots_by" column, further grouped according to the
        "group_by" column.
        If "split_subgroups" is set to true, each group is further split into
        subgroups based on what comes after a potential "-" in the "group_by"
        entry.
        Finally the "metric" entries are averaged to create the bars and the
        standard deviation is shown as error bars.

        Args:
            metrics_df: DataFrame containing the metrics.
            metric: Metric to plot.
            subplots_by: Column to group the subplots by.
            group_by: Column to group the data by.
            split_subgroups: Whether to split subgroups.
        """
        label_replacement_models = {
            "None": "scratch",
            "pret": "pretrained",
            "T": "transfer",
        }

        # Group metrics by the specified columns
        plot_metrics = metrics_df.groupby(metrics_df.columns[subplots_by])

        # Prepare data for plotting
        df_list = []
        for subplot, plot_dfs in plot_metrics:
            group_metrics = plot_dfs.groupby(plot_dfs.columns[group_by])
            for group, group_df in group_metrics:
                if split_subgroups:
                    # Use the shared prefix for subgroups
                    group_split = group.split("-")
                    group = group_split[0]
                    if len(group_split) > 1:
                        subgroup = group_split[1]
                    else:
                        subgroup = "None"
                values = group_df[metric].dropna().astype(float).values
                if values.size == 0:
                    continue
                m, s = np.mean(values), np.std(values)
                df_list.append(
                    {
                        "Subplot": subplot,
                        "Group": group,
                        "Subgroup": subgroup,
                        "Mean": m,
                        "Std": s,
                    }
                )
        df = pd.DataFrame(df_list)
        num_subplots = len(df)
        fig, ax = plt.subplots(
            nrows=num_subplots,
            ncols=1,
            figsize=(self.figsize[0], 0.5 * self.figsize[1] * num_subplots),
        )
        for i, (subplot) in enumerate(df["Subplot"]):
            if num_subplots > 1:
                ax_obj = ax[i]
            else:
                ax_obj = ax
            plot_df = df[df["Subplot"] == subplot].reset_index(drop=True)
            bar_plot = sns.barplot(
                data=plot_df,
                x="Group",
                y="Mean",
                hue="Subgroup",
                errorbar=None,
                ax=ax_obj,
            )
            for i, row in plot_df.iterrows():
                bar_plot.errorbar(
                    i,
                    row["Mean"],
                    yerr=row["Std"],
                    fmt="none",
                    c="black",
                    capsize=3,
                )
            legend_labels = []
            for subgroup in df["Subgroup"].unique():
                if subgroup in label_replacement_models.keys():
                    legend_labels.append(label_replacement_models[subgroup])
                else:
                    legend_labels.append(subgroup)
            handles, _ = ax_obj.get_legend_handles_labels()
            ax_obj.legend(
                handles,
                legend_labels,
                bbox_to_anchor=(0.9, 1),
                loc="upper left",
            )
            ax_obj.set_xlabel("")
            ax_obj.set_ylabel(metric)
            ax_obj.set_title(subplot)
        plt.tight_layout()
        path = os.path.join("plots", "bar_plots")
        self.save_plot(fig, metric, path)

    def _select_top_runs(
        self,
        metric: str,
        metrics: pd.DataFrame,
        metrics_std: pd.DataFrame,
        max_runs: int,
    ):
        if "loss" in metric:
            top_values = metrics.min()
            ascending_order = True
        else:
            m = next((fn for fn in self.metric_fns if fn.name == metric), None)
            if m is None:
                raise ValueError(f"No metric found for '{metric}'.")
            if m.suffix == "max":
                top_values = metrics.max()
                ascending_order = False
            else:
                top_values = metrics.min()
                ascending_order = True

        top_runs = (
            top_values.sort_values(ascending=ascending_order)
            .head(max_runs)
            .index
        )

        metrics = metrics[top_runs]
        if metrics_std is not None:
            metrics_std = metrics_std[top_runs]

        return metrics, metrics_std

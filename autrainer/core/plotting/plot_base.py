from abc import ABC
import logging
import os
import pickle
import shutil

import matplotlib.pyplot as plt
import seaborn as sns


try:
    from latex import escape

    LATEX_AVAILABLE = True
except ImportError:
    LATEX_AVAILABLE = False

    def escape(x):
        return x


class PlotBase(ABC):
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
    ) -> None:
        """Base class for plotting.

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
        """
        self.output_directory = output_directory
        self.training_type = training_type.lower()
        self.figsize = figsize
        self.latex = latex
        self.filetypes = filetypes
        self.pickle = pickle
        self.context = context
        self.palette = palette
        self.replace_none = replace_none
        self.add_titles = add_titles
        self.add_xlabels = add_xlabels
        self.add_ylabels = add_ylabels
        sns.set_theme(
            context=self.context,
            style="whitegrid",
            palette=self.palette,
            rc=rcParams,
        )
        self._log = logging.getLogger(__name__)
        if not self.latex:
            return
        if not LATEX_AVAILABLE:
            self._log.warning(
                "LaTeX is not available as plot style. Install the required "
                "extras with 'pip install autrainer[latex]'."
            )
            self.latex = False
            return
        self._enable_latex()

    def save_plot(
        self,
        fig: plt.Figure,
        name: str,
        path: str = "",
        close: bool = True,
        tight_layout: bool = True,
    ) -> None:
        """Save a plot to the output directory.

        Args:
            fig: Matplotlib figure to save.
            name: Name of the plot.
            path: Path to save the plot to relative to the output directory.
            close: Whether to close the figure after saving.
            tight_layout: Whether to apply tight layout to the plot.
        """
        if self.replace_none:
            self._replace_none(fig)
        if self.latex:
            self._escape_latex(fig)
        base_path = os.path.join(self.output_directory, path)
        os.makedirs(base_path, exist_ok=True)
        if self.pickle:
            pkl_path = os.path.join(base_path, f"{name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(fig, f)
        for filetype in self.filetypes:
            fpath = os.path.join(base_path, f"{name}.{filetype}")
            if tight_layout:
                plt.tight_layout()
            fig.savefig(fpath, dpi=300)
        if close:
            plt.close(fig)

    def _enable_latex(self) -> None:
        if shutil.which("latex") is None:
            self._log.warning(
                "LaTeX is not available on the system. "
                "Install a LaTeX distribution to use LaTeX in plots."
            )
            self.latex = False
            return
        plt.rcParams["text.usetex"] = True

    def _apply_to_labels(self, fig: plt.Figure, func: callable) -> None:
        if not fig.axes or fig.axes[0].legend_ is None:
            return
        for text in fig.axes[0].legend_.texts:
            text.set_text(func(text.get_text()))

    def _escape_latex(self, fig: plt.Figure) -> None:
        self._apply_to_labels(fig, escape)

    def _replace_none(self, fig: plt.Figure) -> None:
        def process_label(label):
            parts = label.split("_")
            parts = [part if part != "None" else "~" for part in parts]
            parts = "_".join(parts).replace("_~", "~").replace("~_", "~")
            return parts

        self._apply_to_labels(fig, process_label)

    def _add_label(
        self, ax: plt.Axes, x_label: str = None, y_label: str = None
    ) -> None:
        if x_label is not None:
            if self.add_xlabels:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel("")
        if y_label is not None:
            if self.add_ylabels:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")

    def _add_titles(self, ax: plt.Axes, title: str) -> None:
        if self.add_titles:
            ax.set_title(title)
        else:
            ax.set_title("")

import os

from hydra_filter_sweeper import AbstractFilter


class EvalFilter(AbstractFilter):
    def filter(self) -> bool:
        training_dir = os.path.join(
            os.path.dirname(os.path.dirname(self.directory)),
            "training",
        )
        run = os.path.basename(self.directory).split("_", 2)[-1]
        return not os.path.exists(os.path.join(training_dir, run, "metrics.csv"))


class TrainFilter(AbstractFilter):
    def filter(self) -> bool:
        return os.path.exists(os.path.join(self.directory, "metrics.csv"))

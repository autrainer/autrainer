import os

from hydra_filter_sweeper import AbstractFilter


class AlreadyRun(AbstractFilter):
    def filter(self) -> bool:
        return os.path.exists(os.path.join(self.directory, "metrics.csv"))

    def reason(self) -> str:
        return "The experiment has already been run."

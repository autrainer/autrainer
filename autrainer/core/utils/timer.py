import datetime
import os
import time

import yaml


class Timer:
    def __init__(self, output_directory: str, timer_type: str) -> None:
        """Timer to measure time of different parts of the training process.

        Args:
            output_directory: Directory to save the timer.yaml file to.
            timer_type: Name of the timer.
        """
        self.time_log = []
        self.start_time = None
        self.output_directory = output_directory
        self.timer_type = timer_type

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop the timer.

        Raises:
            ValueError: If the timer was not started.
        """
        if self.start_time is None:
            raise ValueError("Timer not yet started!")
        run_time = time.time() - self.start_time
        self.start_time = None
        self.time_log.append(run_time)

    def get_time_log(self) -> list:
        """Get the time log.

        Returns:
            List of times.
        """
        return self.time_log

    def get_mean_seconds(self) -> float:
        """Get the mean time in seconds.

        Returns:
            Mean time in seconds.
        """
        return sum(self.time_log) / len(self.time_log)

    def get_total_seconds(self) -> float:
        """Get the total time in seconds.

        Returns:
            Total time in seconds.
        """
        return sum(self.time_log)

    @classmethod
    def pretty_time(cls, seconds: float) -> str:
        """Convert seconds to a pretty string.

        Args:
            seconds: Time in seconds.

        Returns:
            Time in a pretty string format.
        """
        pretty = datetime.timedelta(seconds=int(seconds))
        return str(pretty)

    def save(self, path: str = "") -> dict:
        """Save and append the timer to timer.yaml.

        Args:
            path: Subdirectory to save the timer.yaml file to relative to the
                output directory. Defaults to "".

        Returns:
            Dictionary with mean and total time in seconds and pretty format.
        """
        os.makedirs(os.path.join(self.output_directory, path), exist_ok=True)
        out_path = os.path.join(self.output_directory, path, "timer.yaml")
        time_dict = {
            self.timer_type: {
                "mean": Timer.pretty_time(self.get_mean_seconds()),
                "mean_seconds": self.get_mean_seconds(),
                "total": Timer.pretty_time(self.get_total_seconds()),
                "total_seconds": self.get_total_seconds(),
            }
        }
        with open(out_path, "a") as f:
            yaml.dump(time_dict, f)

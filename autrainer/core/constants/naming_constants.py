from typing import List

from .abstract_constants import AbstractConstants


class NamingConstants(AbstractConstants):
    """Singleton for managing the naming configurations of `autrainer`."""

    _name = "NamingConstants"
    _naming_convention = [
        "dataset",
        "model",
        "optimizer",
        "learning_rate",
        "batch_size",
        "training_type",
        "iterations",
        "scheduler",
        "augmentation",
        "seed",
    ]
    _invalid_aggregations = ["training_type"]
    _valid_aggregations = list(
        set(_naming_convention) - set(_invalid_aggregations)
    )
    _config_dirs = [
        "augmentation",
        "dataset",
        "model",
        "optimizer",
        "plotting",
        "preprocessing",
        "scheduler",
    ]

    @property
    def NAMING_CONVENTION(self) -> List[str]:
        """Get the naming convention of runs.
        Defaults to :attr:`["dataset", "model", "optimizer", "learning_rate",
        "batch_size", "training_type", "iterations", "scheduler",
        "augmentation", "seed"]`.

        Returns:
            Naming convention of runs.
        """
        return self._naming_convention

    @NAMING_CONVENTION.setter
    def NAMING_CONVENTION(self, naming_convention: List[str]) -> None:
        """Set the naming convention of runs.

        Args:
            naming_convention: Naming convention of runs.

        Raises:
            ValueError: If the naming convention is not a list of strings.
        """
        self._assert_type(naming_convention, list, "NAMING_CONVENTION")
        for n in naming_convention:
            self._assert_type(
                n,
                str,
                "NAMING_CONVENTION",
                "in naming convention",
            )
        self._naming_convention = naming_convention

    @property
    def INVALID_AGGREGATIONS(self) -> List[str]:
        """Get the invalid aggregations for postprocessing.
        Defaults to :attr:`["training_type"]`.

        Returns:
            Invalid aggregations for postprocessing.
        """
        return self._invalid_aggregations

    @INVALID_AGGREGATIONS.setter
    def INVALID_AGGREGATIONS(self, invalid_aggregations: List[str]) -> None:
        """Set the invalid aggregations for postprocessing.

        Args:
            invalid_aggregations: Invalid aggregations for postprocessing.

        Raises:
            ValueError: If the invalid aggregations are not a list of strings.
        """
        self._assert_type(invalid_aggregations, list, "INVALID_AGGREGATIONS")
        for i in invalid_aggregations:
            self._assert_type(
                i,
                str,
                "INVALID_AGGREGATIONS",
                "in invalid aggregations",
            )
        self._invalid_aggregations = invalid_aggregations

    @property
    def VALID_AGGREGATIONS(self) -> List[str]:
        """Get the valid aggregations for postprocessing.
        Defaults to :attr:`["dataset", "model", "optimizer", "learning_rate",
        "batch_size", "iterations", "scheduler", "augmentation", "seed"]`
        (the naming convention without the invalid aggregations).

        Returns:
            Valid aggregations for postprocessing.
        """
        return self._valid_aggregations

    @VALID_AGGREGATIONS.setter
    def VALID_AGGREGATIONS(self, valid_aggregations: List[str]) -> None:
        """Set the valid aggregations for postprocessing.

        Args:
            valid_aggregations: Valid aggregations for postprocessing.

        Raises:
            ValueError: If the valid aggregations are not a list of strings.
        """
        self._assert_type(valid_aggregations, list, "VALID_AGGREGATIONS")
        for v in valid_aggregations:
            self._assert_type(
                v,
                str,
                "VALID_AGGREGATIONS",
                "in valid aggregations",
            )
        self._valid_aggregations = valid_aggregations

    @property
    def CONFIG_DIRS(self) -> List[str]:
        """Get the configuration directories for Hydra configurations.
        Defaults to :attr:`["augmentation", "dataset", "model", "optimizer",
        "plotting", "preprocessing", "scheduler"]`.

        Returns:
            Configuration directories for Hydra configurations.
        """
        return self._config_dirs

    @CONFIG_DIRS.setter
    def CONFIG_DIRS(self, config_dirs: List[str]) -> None:
        """Set the configuration directories for Hydra configurations.

        Args:
            config_dirs: Configuration directories for Hydra configurations.

        Raises:
            ValueError: If the configuration directories are not a list of
                strings.
        """
        self._assert_type(config_dirs, list, "CONFIG_DIRS")
        for c in config_dirs:
            self._assert_type(
                c,
                str,
                "CONFIG_DIRS",
                "in configuration directories",
            )
        self._config_dirs = config_dirs

import logging
import os
import random
import time

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest
import torch

from autrainer.core.constants import (
    ExportConstants,
    NamingConstants,
    TrainingConstants,
)
from autrainer.core.utils import (
    Bookkeeping,
    Timer,
    get_hardware_info,
    save_hardware_info,
    set_device,
    set_seed,
    silence,
)
from autrainer.datasets.utils import DataItem
from autrainer.metrics import UAR, Accuracy
from autrainer.models import FFNN

from .utils import BaseIndividualTempDir


class TestBookkeeping(BaseIndividualTempDir):
    def test_log(
        self,
        caplog: pytest.CaptureFixture,
    ) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        with caplog.at_level(logging.INFO):
            bookkeeping.log("test")
            bookkeeping.log("test", level=logging.WARNING)
            bookkeeping.log("test", level=logging.ERROR)
        assert "INFO" in caplog.text, "Should log info message."
        assert "WARNING" in caplog.text, "Should log warning message."
        assert "ERROR" in caplog.text, "Should log error message."

    def test_log_to_file(
        self,
        caplog: pytest.CaptureFixture,
    ) -> None:
        bookkeeping = Bookkeeping(self.temp_dir, "bookkeeping.log")
        with caplog.at_level(logging.INFO):
            bookkeeping.log_to_file("test")
            bookkeeping.log_to_file("test", level=logging.WARNING)
            bookkeeping.log_to_file("test", level=logging.ERROR)
        with open(os.path.join(self.temp_dir, "bookkeeping.log"), "r") as f:
            log = f.read()
        assert not caplog.text, "Should not log to stdout."
        assert "INFO" in log, "Should log info message."
        assert "WARNING" in log, "Should log warning message."
        assert "ERROR" in log, "Should log error message."

    def test_create_folder(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        bookkeeping.create_folder("folder")
        assert os.path.isdir(
            os.path.join(self.temp_dir, "folder")
        ), "Should create folder."

    def test_save_model_summary(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        model = FFNN(output_dim=10, input_size=64, hidden_size=64)
        dataset = [DataItem(features=torch.randn(64), target=None, index=None)]
        bookkeeping.save_model_summary(model, dataset, "summary.txt")
        assert os.path.isfile(
            os.path.join(self.temp_dir, "summary.txt")
        ), "Should save model summary txt."
        assert os.path.isfile(
            os.path.join(self.temp_dir, "summary.yaml")
        ), "Should save model summary YAML."

    def test_save_state_dict(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        model = FFNN(output_dim=10, input_size=64, hidden_size=64)
        bookkeeping.save_state(model, "model.pt")
        assert os.path.isfile(
            os.path.join(self.temp_dir, "model.pt")
        ), "Should save model state dict."

    def test_save_invalid_state_dict(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        model = "not a model"
        with pytest.raises(TypeError):
            bookkeeping.save_state(model, "model.pt")
        assert not os.path.isfile(
            os.path.join(self.temp_dir, "model.pt")
        ), "Should not save model state dict."

    def test_load_state_dict(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        model1 = FFNN(output_dim=10, input_size=64, hidden_size=64)
        bookkeeping.save_state(model1, "model.pt")
        model2 = FFNN(output_dim=10, input_size=64, hidden_size=64)
        bookkeeping.load_state(model2, "model.pt")
        assert str(model1.state_dict()) == str(
            model2.state_dict()
        ), "Should load model state dict."

    def test_load_invalid_state_dict(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        model = FFNN(output_dim=10, input_size=64, hidden_size=64)
        bookkeeping.save_state(model, "model.pt")
        model = "not a model"
        with pytest.raises(TypeError):
            bookkeeping.load_state(model, "model.pt")

    def test_load_non_existent_state_dict(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        model = FFNN(output_dim=10, input_size=64, hidden_size=64)
        with pytest.raises(FileNotFoundError):
            bookkeeping.load_state(model, "model.pt")

    def test_save_audobject(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        model = FFNN(output_dim=10, input_size=64, hidden_size=64)
        bookkeeping.save_audobject(model, "model.yaml")
        assert os.path.isfile(
            os.path.join(self.temp_dir, "model.yaml")
        ), "Should save model audobject YAML."

    def test_save_invalid_audobject(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        model = "not a model"
        with pytest.raises(TypeError):
            bookkeeping.save_audobject(model, "model.yaml")
        assert not os.path.isfile(
            os.path.join(self.temp_dir, "model.yaml")
        ), "Should not save model audobject YAML."

    def test_save_results_dict(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        results = {"loss": 0.1, "accuracy": 0.9}
        bookkeeping.save_results_dict(results, "results.yaml")
        assert os.path.isfile(
            os.path.join(self.temp_dir, "results.yaml")
        ), "Should save results dict YAML."
        loaded = OmegaConf.load(os.path.join(self.temp_dir, "results.yaml"))
        loaded = OmegaConf.to_container(loaded)
        assert results == loaded, "Should save and load results dict YAML."

    def test_save_results_df(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        results = pd.DataFrame({"loss": [0.1], "accuracy": [0.9]})
        bookkeeping.save_results_df(results, "results.csv")
        assert os.path.isfile(
            os.path.join(self.temp_dir, "results.csv")
        ), "Should save results dict CSV."
        loaded = pd.read_csv(os.path.join(self.temp_dir, "results.csv"))
        assert results.equals(loaded), "Should save and load results dict CSV."

    def test_save_results_np(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        results = np.array([[0.1, 0.9]])
        bookkeeping.save_results_np(results, "results.npy")
        assert os.path.isfile(
            os.path.join(self.temp_dir, "results.npy")
        ), "Should save results dict NPY."
        loaded = np.load(os.path.join(self.temp_dir, "results.npy"))
        (
            np.testing.assert_array_equal(results, loaded),
            "Should save and load results dict NPY.",
        )

    def test_save_best_results(self) -> None:
        bookkeeping = Bookkeeping(self.temp_dir)
        results = pd.DataFrame(
            {
                "train_loss": [0.01, 0.1, 0.2],
                "accuracy": [0.9, 0.99, 0.8],
                "uar": [0.8, 0.7, 0.9],
            }
        )
        metrics = [Accuracy(), UAR()]
        tracking_metric = Accuracy()
        bookkeeping.save_best_results(
            metrics=results,
            filename="best_results.yaml",
            metric_fns=metrics,
            tracking_metric_fn=tracking_metric,
        )
        assert os.path.isfile(
            os.path.join(self.temp_dir, "best_results.yaml")
        ), "Should save best results dict YAML."
        loaded = OmegaConf.load(
            os.path.join(self.temp_dir, "best_results.yaml")
        )
        loaded = OmegaConf.to_container(loaded)
        assert loaded.get("best_iteration") == 1, "Should save best iteration."
        assert loaded.get("accuracy") == 0.99, "Should save best accuracy."
        assert loaded.get("uar") == 0.9, "Should save best UAR."
        assert (
            loaded.get("train_loss_min") == 0.01
        ), "Should save best train loss."


class TestTimer(BaseIndividualTempDir):
    def test_timer(self) -> None:
        timer = Timer(self.temp_dir, "test")
        timer.start()
        time.sleep(1)
        timer.stop()
        timer.save()
        assert timer.get_time_log(), "Should get time log."
        assert timer.get_mean_seconds(), "Should get mean time in seconds."
        assert timer.get_total_seconds(), "Should get total time in seconds."
        assert timer.pretty_time(1), "Should convert seconds to pretty string."
        assert os.path.isfile(
            os.path.join(self.temp_dir, "timer.yaml")
        ), "Should save timer."

    def test_invalid_timer(self) -> None:
        timer = Timer(self.temp_dir, "test")
        with pytest.raises(ValueError):
            timer.stop()


class TestHardware(BaseIndividualTempDir):
    @classmethod
    def setup_class(cls) -> None:
        cls.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def test_get_hardware_info(self) -> dict:
        hardware_info = get_hardware_info(self.device)
        assert hardware_info, "Should get hardware info."

    def test_save_hardware_info(self) -> None:
        save_hardware_info(self.temp_dir, device=self.device)
        assert os.path.isfile(
            os.path.join(self.temp_dir, "hardware.yaml")
        ), "Should save hardware info."

    def test_set_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = set_device("cuda:0")
            assert device.type == "cuda", "Should set CUDA device."
        device = set_device("cpu")
        assert device.type == "cpu", "Should set CPU device."

    def test_fallback_device(self, caplog: pytest.CaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            device = set_device("invalid")
        assert device.type == "cpu", "Should fall back to CPU."
        assert (
            "Device 'invalid' is not available. Falling back to CPU."
            in caplog.text
        ), "Should log warning."


class TestSetSeed:
    @pytest.mark.parametrize("seed", [42, 0])
    def test_set_seed(self, seed: int) -> None:
        set_seed(seed)
        rand = random.random()
        random.seed(seed)

        assert random.random() == rand, "Should set Python seed."
        assert np.random.get_state()[1][0] == seed, "Should set NumPy seed."
        assert torch.initial_seed() == seed, "Should set Torch seed."
        if torch.cuda.is_available():
            assert torch.cuda.initial_seed() == seed, "Should set Cuda seed."
            assert (
                torch.backends.cudnn.deterministic
            ), "Should be deterministic."
            assert (
                not torch.backends.cudnn.benchmark
            ), "Should disable benchmark."


class TestSilence:
    def test_silence(self, capfd: pytest.CaptureFixture) -> None:
        with silence():
            print("Should not print.")
        print("Should print.")
        out, _ = capfd.readouterr()
        assert "Should print." in out, "Should print."
        assert "Should not print." not in out, "Should not print."


class TestExportConstants:
    @classmethod
    def setup_class(cls) -> None:
        cls.c = ExportConstants()
        cls._logging_depth = cls.c.LOGGING_DEPTH
        cls._ignore_params = cls.c.IGNORE_PARAMS.copy()
        cls._artifacts = cls.c.ARTIFACTS.copy()

    @classmethod
    def teardown_class(cls) -> None:
        cls.c.LOGGING_DEPTH = cls._logging_depth
        cls.c.IGNORE_PARAMS = cls._ignore_params
        cls.c.ARTIFACTS = cls._artifacts

    @pytest.mark.parametrize("depth", [-1, "invalid"])
    def test_invalid_logging_depth(self, depth: int) -> None:
        with pytest.raises(ValueError):
            self.c.LOGGING_DEPTH = depth

    @pytest.mark.parametrize("depth", [1, 2, 3, 100])
    def test_logging_depth(self, depth: int) -> None:
        self.c.LOGGING_DEPTH = depth
        assert (
            ExportConstants().LOGGING_DEPTH == depth
        ), f"Should set logging depth to {depth}."

    @pytest.mark.parametrize("ignore_params", ["invalid", [1, "param"]])
    def test_invalid_ignore_params(self, ignore_params: list) -> None:
        with pytest.raises(ValueError):
            self.c.IGNORE_PARAMS = ignore_params

    @pytest.mark.parametrize("ignore_params", [[], ["param1", "param2"]])
    def test_ignore_params(self, ignore_params: list) -> None:
        self.c.IGNORE_PARAMS = ignore_params
        assert (
            ExportConstants().IGNORE_PARAMS == ignore_params
        ), f"Should set ignore params to {ignore_params}."

    @pytest.mark.parametrize(
        "artifacts",
        [[1, "artifact"], ["artifact", {"invalid": 1}]],
    )
    def test_invalid_artifacts(self, artifacts: list) -> None:
        with pytest.raises(ValueError):
            self.c.ARTIFACTS = artifacts

    @pytest.mark.parametrize(
        "artifacts",
        [["artifact"], ["artifact", {"artifact": "artifact"}]],
    )
    def test_artifacts(self, artifacts: list) -> None:
        self.c.ARTIFACTS = artifacts
        assert (
            ExportConstants().ARTIFACTS == artifacts
        ), f"Should set artifacts to {artifacts}."


class TestNamingConstants:
    @classmethod
    def setup_class(cls) -> None:
        cls.c = NamingConstants()
        cls._config_dirs = cls.c.CONFIG_DIRS.copy()
        cls._naming_convention = cls.c.NAMING_CONVENTION.copy()
        cls._valid_aggregations = cls.c.VALID_AGGREGATIONS.copy()
        cls._invalid_aggregations = cls.c.INVALID_AGGREGATIONS.copy()

    @classmethod
    def teardown_class(cls) -> None:
        cls.c.CONFIG_DIRS = cls._config_dirs
        cls.c.NAMING_CONVENTION = cls._naming_convention
        cls.c.VALID_AGGREGATIONS = cls._valid_aggregations
        cls.c.INVALID_AGGREGATIONS = cls._invalid_aggregations

    @pytest.mark.parametrize("config_dirs", ["invalid", [1, "invalid"]])
    def test_invalid_config_dirs(self, config_dirs: list) -> None:
        with pytest.raises(ValueError):
            self.c.CONFIG_DIRS = config_dirs

    @pytest.mark.parametrize("config_dirs", [["dir1", "dir2"]])
    def test_config_dirs(self, config_dirs: list) -> None:
        self.c.CONFIG_DIRS = config_dirs
        assert (
            NamingConstants().CONFIG_DIRS == config_dirs
        ), f"Should set config dirs to {config_dirs}."

    @pytest.mark.parametrize("naming_convention", ["invalid", [1, "invalid"]])
    def test_invalid_naming_convention(self, naming_convention: list) -> None:
        with pytest.raises(ValueError):
            self.c.NAMING_CONVENTION = naming_convention

    @pytest.mark.parametrize("naming_convention", [["dir1", "dir2"]])
    def test_naming_convention(self, naming_convention: list) -> None:
        self.c.NAMING_CONVENTION = naming_convention
        assert (
            NamingConstants().NAMING_CONVENTION == naming_convention
        ), f"Should set naming convention to {naming_convention}."

    @pytest.mark.parametrize(
        "valid_aggregations",
        ["invalid", [1, "invalid"]],
    )
    def test_invalid_valid_aggregations(
        self, valid_aggregations: list
    ) -> None:
        with pytest.raises(ValueError):
            self.c.VALID_AGGREGATIONS = valid_aggregations

    @pytest.mark.parametrize(
        "valid_aggregations",
        [["model", "dataset"]],
    )
    def test_valid_aggregations(self, valid_aggregations: list) -> None:
        self.c.VALID_AGGREGATIONS = valid_aggregations
        assert (
            NamingConstants().VALID_AGGREGATIONS == valid_aggregations
        ), f"Should set valid aggregations to {valid_aggregations}."

    @pytest.mark.parametrize(
        "invalid_aggregations",
        ["invalid", [1, "invalid"]],
    )
    def test_invalid_invalid_aggregations(
        self, invalid_aggregations: list
    ) -> None:
        with pytest.raises(ValueError):
            self.c.INVALID_AGGREGATIONS = invalid_aggregations

    @pytest.mark.parametrize(
        "invalid_aggregations",
        [["model", "dataset"]],
    )
    def test_invalid_aggregations(self, invalid_aggregations: list) -> None:
        self.c.INVALID_AGGREGATIONS = invalid_aggregations
        assert (
            NamingConstants().INVALID_AGGREGATIONS == invalid_aggregations
        ), f"Should set invalid aggregations to {invalid_aggregations}."


class TestTrainingConstants:
    @classmethod
    def setup_class(cls) -> None:
        cls.c = TrainingConstants()
        cls._tasks = cls.c.TASKS.copy()

    @classmethod
    def teardown_class(cls) -> None:
        cls.c.TASKS = cls._tasks

    @pytest.mark.parametrize("tasks", ["invalid", [1, "invalid"]])
    def test_invalid_tasks(self, tasks: list) -> None:
        with pytest.raises(ValueError):
            self.c.TASKS = tasks

    @pytest.mark.parametrize("tasks", [["task1", "task2"]])
    def test_tasks(self, tasks: list) -> None:
        self.c.TASKS = tasks
        assert (
            TrainingConstants().TASKS == tasks
        ), f"Should set tasks to {tasks}."

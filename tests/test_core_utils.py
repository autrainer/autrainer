import logging
import os
import random
import time

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest
import torch

from autrainer.core.utils import (
    Bookkeeping,
    Timer,
    get_hardware_info,
    save_hardware_info,
    set_seed,
    silence,
)
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
        dataset = [[torch.randn(64)]]
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


class TestHardwareInfo(BaseIndividualTempDir):
    def test_get_hardware_info(self) -> dict:
        hardware_info = get_hardware_info()
        assert hardware_info, "Should get hardware info."

    def test_save_hardware_info(self) -> None:
        save_hardware_info(self.temp_dir)
        assert os.path.isfile(
            os.path.join(self.temp_dir, "hardware.yaml")
        ), "Should save hardware info."


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

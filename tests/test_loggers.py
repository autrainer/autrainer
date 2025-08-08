import logging
import os
import time
from typing import Union

from omegaconf import OmegaConf
import pytest

from autrainer.core.utils import Timer
from autrainer.loggers import FallbackLogger, MLFlowLogger, TensorBoardLogger
from autrainer.metrics import UAR, Accuracy

from .utils import BaseIndividualTempDir


class TestFallbackLogger:
    @pytest.mark.parametrize("requested", ["mlflow", "test", None])
    def test_requested(
        self,
        requested: Union[str, None],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            FallbackLogger(requested, None)
        if requested is None:
            return
        assert f"Requested logger '{requested}' not available." in caplog.text, (
            "Should log warning"
        )

    def test_identity_functions(self) -> None:
        logger = FallbackLogger()
        logger.log_params({})
        logger.log_and_update_metrics({})
        logger.log_metrics({})
        logger.log_timers({})
        logger.log_artifact("")


class TestMLFlowLogger(BaseIndividualTempDir):
    @staticmethod
    def _mock_mlflow_logger(
        exp_name: str,
        run_name: str,
    ) -> MLFlowLogger:
        logger = MLFlowLogger(
            exp_name=exp_name,
            run_name=run_name,
            metrics=[Accuracy(), UAR()],
            tracking_metric=Accuracy(),
            output_dir="mlruns",
        )
        logger.setup()
        return logger

    @staticmethod
    def _file_exists(name: str, log_type: str) -> bool:
        for root, dirs, files in os.walk("mlruns"):
            if log_type in root:
                if name in files:
                    return True
        return False

    def test_setup_and_multiple_runs(self) -> None:
        logger = self._mock_mlflow_logger("test_exp", "run1")
        assert logger.exp_id is not None, "Should create experiment"
        assert logger.run is not None, "Should create run"
        logger.end_run()

        new_logger = self._mock_mlflow_logger("test_exp", "run1")
        new_logger.end_run()
        assert logger.exp_id == new_logger.exp_id, "Should use same experiment"
        new_logger.end_run()

    def test_log_params(self) -> None:
        logger = self._mock_mlflow_logger("test_exp", "run1")
        logger.log_params({"a": 1, "b": 2})
        logger.end_run()
        assert self._file_exists("a", "params"), "Should log params"
        assert self._file_exists("b", "params"), "Should log params"

    def test_log_metrics(self) -> None:
        logger = self._mock_mlflow_logger("test_exp", "run1")
        logger.log_metrics({"accuracy": 0.5, "uar": 0.6}, iteration=1)
        logger.log_metrics({"accuracy": 0.8, "uar": 0.7}, iteration=2)
        logger.end_run()
        assert self._file_exists("accuracy", "metrics"), "Should log metrics"
        assert self._file_exists("uar", "metrics"), "Should log metrics"

    def test_log_timers(self) -> None:
        logger = self._mock_mlflow_logger("test_exp", "run1")
        timer = Timer("mlruns", "test")
        timer.start()
        timer.stop()
        logger.log_timers({"time.test.mean": timer.get_mean_seconds()})
        logger.end_run()
        assert self._file_exists("time.test.mean", "params"), "Should log timers"

    def test_log_artifact(self) -> None:
        OmegaConf.save({"model_parameters": 42}, "model_summary.yaml")
        logger = self._mock_mlflow_logger("test_exp", "run1")
        logger.log_artifact("model_summary.yaml")
        logger.end_run()
        assert self._file_exists("model_summary.yaml", "artifacts")


class TestTensorBoardLogger(BaseIndividualTempDir):
    @staticmethod
    def _mock_tensorboard_logger(
        exp_name: str,
        run_name: str,
    ) -> MLFlowLogger:
        logger = TensorBoardLogger(
            exp_name=exp_name,
            run_name=run_name,
            metrics=[Accuracy(), UAR()],
            tracking_metric=Accuracy(),
            output_dir="tbruns",
        )
        logger.setup()
        return logger

    def test_identity(self) -> None:
        logger = self._mock_tensorboard_logger("test_exp", "run1")
        logger.log_artifact("model_summary.yaml")  # unable to log artifacts
        logger.end_run()

    def test_log_params(self) -> None:
        logger = self._mock_tensorboard_logger("test_exp", "run1")
        logger.log_params({"a": 1, "b": 2})
        logger.end_run()
        assert len(os.listdir("tbruns/test_exp/run1")) > 0, "Should log params"

    def test_log_metrics(self) -> None:
        logger = self._mock_tensorboard_logger("test_exp", "run1")
        logger.log_metrics({"accuracy": 0.5, "uar": 0.6}, iteration=1)
        logger.end_run()
        assert len(os.listdir("tbruns/test_exp/run1")) > 0, "Should log metrics"

    def test_log_timers(self) -> None:
        logger = self._mock_tensorboard_logger("test_exp", "run1")
        timer = Timer("tbruns", "test")
        timer.start()
        timer.stop()
        logger.log_timers({"time.test.mean": timer.get_mean_seconds()})
        logger.end_run()
        assert len(os.listdir("tbruns/test_exp/run1")) > 0, "Should log timers"


class TestSimulateTraining(BaseIndividualTempDir):
    class MockTrainer:
        cfg = {
            "model": {"id": "TestModel", "_target_": "some.Model"},
            "augmentation": {"id": "None"},
            "dataset": {
                "id": "TestDataset",
                "_target_": "some.Dataset",
                "transform": {"TestTransform": {"transform": "value"}},
            },
            "iterations": 100,
            "_ignore_": "value",
        }
        train_timer = Timer("tbruns", "train")
        dev_timer = Timer("tbruns", "dev")
        test_timer = Timer("tbruns", "test")
        output_directory = ""

    @staticmethod
    def _mock_trainer() -> MockTrainer:
        m = TestSimulateTraining.MockTrainer()
        os.makedirs(".hydra", exist_ok=True)
        OmegaConf.save({}, "model_summary.txt")
        OmegaConf.save({}, "metrics.csv")
        OmegaConf.save(m.cfg, ".hydra/config.yaml")
        for timer in [m.train_timer, m.dev_timer, m.test_timer]:
            timer.start()
        time.sleep(1)
        for timer in [m.train_timer, m.dev_timer, m.test_timer]:
            timer.stop()
        return m

    def test_simulate_training(self) -> None:
        logger = TensorBoardLogger(
            exp_name="test_exp",
            run_name="test_run",
            metrics=[Accuracy(), UAR()],
            tracking_metric=Accuracy(),
            output_dir="tbruns",
        )
        logger.setup()
        trainer = self._mock_trainer()
        logger.cb_on_train_begin(trainer)
        for i in range(5):
            logger.cb_on_iteration_end(
                trainer,
                i,
                {
                    "accuracy": 0.1 + (0.1 * i),
                    "uar": 0.2 + (0.1 * i),
                    "dev_loss": 2.3 - (0.1 * i),
                },
            )

        logger.cb_on_test_end(
            trainer,
            {"accuracy": 0.5, "uar": 0.6, "dev_loss": 1.0},
        )
        logger.cb_on_train_end(trainer)

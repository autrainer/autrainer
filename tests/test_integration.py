import os

from omegaconf import DictConfig, OmegaConf
import pytest

import autrainer.cli

from .utils import BaseIndividualTempDir


class TestCLIIntegration(BaseIndividualTempDir):
    @pytest.mark.parametrize(
        "dataset, model, train_type, iterations, reuse_temp_dir",
        [
            ("ToyTabular-C", "ToyFFNN", "epoch", 1, False),
            ("ToyTabular-C", "ToyFFNN", "epoch", 2, True),
            ("ToyTabular-R", "ToyFFNN", "epoch", 1, False),
            ("ToyTabular-MLC", "ToyFFNN", "epoch", 1, False),
            ("ToyTabular-C", "ToyFFNN", "step", 100, False),
            ("ToyTabular-C", "ToyFFNN", "step", 200, True),
            ("ToyTabular-R", "ToyFFNN", "step", 100, False),
            ("ToyTabular-MLC", "ToyFFNN", "step", 100, False),
        ],
    )
    def test_train_postprocess(
        self,
        dataset: str,
        model: str,
        train_type: str,
        iterations: int,
        reuse_temp_dir: bool,
        capfd: pytest.CaptureFixture,
    ) -> None:
        assert train_type in ["epoch", "step"], "Invalid train type."
        autrainer.cli.create(empty=True, force=True)
        config = OmegaConf.load("conf/config.yaml")
        if train_type == "epoch":
            config = self._setup_epoch(config, iterations)
        else:
            config = self._setup_step(config, iterations)
        config["progress_bar"] = False
        config["hydra"]["sweeper"]["params"]["dataset"] = dataset
        config["hydra"]["sweeper"]["params"]["model"] = model
        OmegaConf.save(config, "conf/config.yaml")
        self._train_postprocess(capfd)

    @staticmethod
    def _setup_epoch(config: DictConfig, iterations: int) -> DictConfig:
        config["iterations"] = iterations
        return config

    @staticmethod
    def _setup_step(config: DictConfig, iterations: int) -> DictConfig:
        config["training_type"] = "step"
        config["iterations"] = iterations
        config["eval_frequency"] = 100
        config["save_frequency"] = iterations
        return config

    @staticmethod
    def _train_postprocess(capfd: pytest.CaptureFixture) -> None:
        autrainer.cli.train()
        out, _ = capfd.readouterr()
        print(out)
        assert "Best results at" in out, "Should print training results."
        assert "Test results:" in out, "Should print test results."

        autrainer.cli.train()
        out, _ = capfd.readouterr()
        assert "Filtered:" in out, "Should print skipping message."

        autrainer.cli.postprocess("results", "default")
        autrainer.cli.postprocess("results", "default", aggregate=[["seed"]])
        _, err = capfd.readouterr()
        assert err == "", "Should not print error message."
        run_dirs = [
            f
            for f in os.listdir("results/default/training")
            if os.path.isdir(os.path.join("results/default/training", f))
        ]
        assert len(run_dirs) == 1, "Should have only one run directory."

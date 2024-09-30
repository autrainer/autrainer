import os
from unittest.mock import patch
import warnings

from omegaconf import DictConfig, OmegaConf
import pytest
import torch

import autrainer.cli
from autrainer.models import FFNN

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
        with patch("sys.argv", [""]):
            autrainer.cli.train()
        out, _ = capfd.readouterr()
        print(out)
        assert "Best results at" in out, "Should print training results."
        assert "Test results:" in out, "Should print test results."

        with patch("sys.argv", [""]):
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


class TestCheckpointsIntegration(BaseIndividualTempDir):
    @staticmethod
    def _load_state(module: torch.nn.Module, name: str, subdir: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            module.load_state_dict(
                torch.load(
                    f"results/default/training/{name}/{subdir}/model.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
        module.eval()

    def test_checkpoints(self) -> None:
        autrainer.cli.create(empty=True)
        with patch("sys.argv", [""]):
            autrainer.cli.train({"scheduler": "StepLR"})
        run_dirs = [
            f
            for f in os.listdir("results/default/training")
            if os.path.isdir(os.path.join("results/default/training", f))
        ]
        assert len(run_dirs) == 1, "Should have only one run directory."

        autrainer.cli.show("model", "ToyFFNN", save=True)
        model_cfg = OmegaConf.load("conf/model/ToyFFNN.yaml")
        run_dir = f"results/default/training/{run_dirs[0]}/_best"
        model_cfg["id"] = "ToyFFNN-CKPT"
        model_cfg["model_checkpoint"] = f"{run_dir}/model.pt"
        model_cfg["optimizer_checkpoint"] = f"{run_dir}/optimizer.pt"
        model_cfg["scheduler_checkpoint"] = f"{run_dir}/scheduler.pt"
        OmegaConf.save(model_cfg, "conf/model/ToyFFNN-CKPT.yaml")

        autrainer.cli.show("dataset", "ToyTabular-C", save=True)
        dataset_cfg = OmegaConf.load("conf/dataset/ToyTabular-C.yaml")
        dataset_cfg["id"] = "ToyTabular-C-7"
        dataset_cfg["num_targets"] = 7
        OmegaConf.save(dataset_cfg, "conf/dataset/ToyTabular-C-7.yaml")

        with patch("sys.argv", [""]):
            autrainer.cli.train(
                {
                    "dataset": "ToyTabular-C-7",
                    "model": "ToyFFNN-CKPT",
                    "scheduler": "StepLR",
                }
            )
        run_dirs = [
            f
            for f in os.listdir("results/default/training")
            if os.path.isdir(os.path.join("results/default/training", f))
        ]
        assert len(run_dirs) == 2, "Should have two run directories."
        r1 = next(r for r in run_dirs if "ToyFFNN-CKPT" not in r)
        r2 = next(r for r in run_dirs if "ToyFFNN-CKPT" in r)
        m1 = FFNN(10, 64, 64)
        self._load_state(m1, r1, "_best")
        m2 = FFNN(7, 64, 64)
        self._load_state(m2, r2, "_initial")
        x = torch.randn(4, 64)
        assert torch.allclose(
            m1.embeddings(x), m2.embeddings(x)
        ), "Should have same embeddings."

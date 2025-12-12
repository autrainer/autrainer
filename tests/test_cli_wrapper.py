import os
import subprocess
from typing import List, Union
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest

import autrainer.cli
from autrainer.core.constants import NamingConstants
from autrainer.core.scripts.command_line_error import CommandLineError

from .utils import BaseIndividualTempDir


class TestMainEntryPoint(BaseIndividualTempDir):
    @pytest.mark.parametrize(
        "args",
        [
            ["-h"],
            ["-v"],
            ["create", "-e"],
        ],
    )
    def test_main(self, args: List[str]) -> None:
        result = subprocess.run(
            ["autrainer"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 0, "Should return 0."
        assert result.stderr == "", "Should not return an error message."

    def test_no_command(self) -> None:
        result = subprocess.run(
            ["autrainer"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 2, "Should return 2."

    def test_non_existent_command(self) -> None:
        result = subprocess.run(
            ["autrainer", "create", "invalid"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 2, "Should return 2."


class TestCLICreate(BaseIndividualTempDir):
    def test_empty_directory(self) -> None:
        with pytest.raises(
            CommandLineError,
            match="No configuration directories specified.",
        ):
            autrainer.cli.create([])

    def test_invalid_directory(self) -> None:
        with pytest.raises(
            CommandLineError,
            match="Invalid configuration directory 'invalid'.",
        ):
            autrainer.cli.create(["invalid"])

    @pytest.mark.parametrize(
        ("dirs", "empty", "all"),
        [
            (None, True, True),
            (["model"], False, True),
            (["model"], True, False),
        ],
    )
    def test_mutually_exclusive(
        self,
        dirs: Union[List[str]],
        empty: bool,
        all: bool,
    ) -> None:
        with pytest.raises(
            CommandLineError,
            match="The flags -e/--empty and -a/--all are mutually exclusive",
        ):
            autrainer.cli.create(dirs, empty, all)

    def test_force_overwrite(self) -> None:
        os.mkdir("conf")
        with pytest.raises(
            CommandLineError,
            match="Directory 'conf' already exists.",
        ):
            autrainer.cli.create(empty=True)
        autrainer.cli.create(empty=True, force=True)

    @pytest.mark.parametrize(
        "dirs",
        [
            ["model"],
            ["model", "dataset"],
            ["model", "dataset", "optimizer", "scheduler"],
            NamingConstants().CONFIG_DIRS,
        ],
    )
    def test_create_directories(self, dirs: List[str]) -> None:
        autrainer.cli.create(dirs)
        assert all(os.path.exists(f"conf/{directory}") for directory in dirs), (
            "Should create directories."
        )
        assert os.path.exists("conf/config.yaml"), "Should create config.yaml."

    def test_create_empty(self) -> None:
        autrainer.cli.create(empty=True)
        assert os.path.exists("conf/config.yaml"), "Should create config.yaml."
        assert os.listdir("conf") == ["config.yaml"], "Should only contain config.yaml."

    def test_create_eval(self) -> None:
        autrainer.cli.create(empty=True, eval=True)
        for file in ["config.yaml", "eval.yaml"]:
            assert os.path.exists(f"conf/{file}"), f"Should create {file}."
        assert set(os.listdir("conf")) == {"config.yaml", "eval.yaml"}, (
            "Should only contain config.yaml and eval.yaml."
        )

    def test_create_all(self) -> None:
        autrainer.cli.create(all=True)
        assert all(
            os.path.exists(f"conf/{directory}")
            for directory in NamingConstants().CONFIG_DIRS
        ), "Should create all directories."
        for file in ["config.yaml", "eval.yaml"]:
            assert os.path.exists(f"conf/{file}"), f"Should create {file}."


class TestCLIList(BaseIndividualTempDir):
    def test_invalid_directory(self) -> None:
        with pytest.raises(
            CommandLineError,
            match="Invalid configuration directory 'invalid'.",
        ):
            autrainer.cli.list("invalid")

    def test_local_invalid_directory(self) -> None:
        with pytest.raises(
            CommandLineError,
            match="Local conf directory 'model' does not exist.",
        ):
            autrainer.cli.list("model", local_only=True)

    @pytest.mark.parametrize(
        ("local_only", "global_only"),
        [(True, False), (False, True), (True, True)],
    )
    def test_local_global_configs(
        self,
        capfd: pytest.CaptureFixture,
        local_only: bool,
        global_only: bool,
    ) -> None:
        os.makedirs("conf/model", exist_ok=True)
        OmegaConf.save({}, "conf/model/EfficentNet-B42.yaml")
        autrainer.cli.list(
            "model",
            local_only=local_only,
            global_only=global_only,
        )
        out, _ = capfd.readouterr()
        if local_only:
            assert "Local 'model' configurations:" in out, (
                "Should print local configurations."
            )
        if global_only:
            assert "Global 'model' configurations:" in out, (
                "Should print global configurations."
            )

    def test_local_global_missing_configs(self, capfd: pytest.CaptureFixture) -> None:
        os.makedirs("conf/model", exist_ok=True)
        autrainer.cli.list("model", pattern="MissingNet-*")
        out, _ = capfd.readouterr()
        assert "No local 'model' configurations found." in out, (
            "Should not print local configurations."
        )
        assert "No global 'model' configurations found." in out, (
            "Should not print global configurations."
        )


class TestCLIShow(BaseIndividualTempDir):
    @pytest.mark.parametrize(
        "config",
        ["EfficientNet-B0", "EfficientNet-B0.yaml"],
    )
    def test_valid_directory(self, capfd: pytest.CaptureFixture, config: str) -> None:
        autrainer.cli.show("model", config)
        out, _ = capfd.readouterr()
        config = config.replace(".yaml", "")
        assert f"id: {config}" in out, "Should print configuration."

    def test_invalid_directory(self) -> None:
        with pytest.raises(
            CommandLineError,
            match="Invalid configuration directory 'invalid'.",
        ):
            autrainer.cli.list("invalid")

    def test_invalid_config(self) -> None:
        with pytest.raises(
            CommandLineError,
            match=r"No global configuration 'InvalidNet' found for 'model'.",
        ):
            autrainer.cli.show("model", "InvalidNet")

    def test_save_configuration(self, capfd: pytest.CaptureFixture) -> None:
        autrainer.cli.show("model", "EfficientNet-B0", save=True)
        out, _ = capfd.readouterr()
        assert "id: EfficientNet-B0" in out, "Should print configuration."
        assert os.path.exists("conf/model/EfficientNet-B0.yaml"), (
            "Should save configuration."
        )

    def test_force_overwrite(self, capfd: pytest.CaptureFixture) -> None:
        os.makedirs("conf/model", exist_ok=True)
        OmegaConf.save({}, "conf/model/EfficientNet-B0.yaml")
        with pytest.raises(
            CommandLineError,
            match=".+'EfficientNet-B0' already exists.+",
        ):
            autrainer.cli.show("model", "EfficientNet-B0", save=True)

        autrainer.cli.show("model", "EfficientNet-B0", save=True, force=True)
        out, _ = capfd.readouterr()
        assert "id: EfficientNet-B0" in out, "Should print configuration."
        assert os.path.exists("conf/model/EfficientNet-B0.yaml"), (
            "Should save configuration."
        )


class TestCLIFetch(BaseIndividualTempDir):
    @pytest.mark.parametrize("cfg_launcher", [False, True])
    def test_launcher_override(
        self, capfd: pytest.CaptureFixture, cfg_launcher: bool
    ) -> None:
        with patch("sys.argv", [""]):
            autrainer.cli.fetch(
                cfg_launcher=cfg_launcher,
            )
        out, _ = capfd.readouterr()
        assert "Fetching datasets..." in out, "Should print fetching message."
        assert "Fetching models..." in out, "Should print fetching message."

    def test_notebook(self, capfd: pytest.CaptureFixture) -> None:
        with patch("sys.argv", ["ipykernel"]):
            autrainer.cli.fetch()
        out, _ = capfd.readouterr()
        assert "Fetching datasets..." in out, "Should print fetching message."
        assert "Fetching models..." in out, "Should print fetching message."


class TestCLIPreprocess(BaseIndividualTempDir):
    def test_preprocess(self, capfd: pytest.CaptureFixture) -> None:
        with patch("sys.argv", [""]):
            autrainer.cli.preprocess(cfg_launcher=True, update_frequency=0)
        out, _ = capfd.readouterr()
        assert "Preprocessing datasets..." in out, "Should print preprocessing message."


class TestCLIInference(BaseIndividualTempDir):
    # inference run is covered in test_serving.py
    def _mock_model(self) -> None:
        os.makedirs("model", exist_ok=True)
        for file in [
            "model.yaml",
            "file_handler.yaml",
            "target_transform.yaml",
            "inference_transform.yaml",
            "preprocess_file_handler.yaml",
            "preprocess_pipeline.yaml",
        ]:
            OmegaConf.save({}, f"model/{file}")

    def _mock_input(self) -> None:
        os.makedirs("input", exist_ok=True)
        OmegaConf.save({}, "input/file.wav")

    @pytest.mark.parametrize(
        "model",
        [
            "",
            "invalid_local",
            "hf:invalid",
            "hf:",
            "hf:user_id/",
            "hf:user_id/repo_id",
            "hf:user_id/repo_id@main",
            "hf:user_id/repo_id@main:subdir",
            "hf:user_id/repo_id@main:subdir#local_dir",
        ],
    )
    def test_invalid_model(self, model: str) -> None:
        with pytest.raises(CommandLineError, match="Invalid"):
            autrainer.cli.inference(model=model, input="", output="")

    def test_invalid_input(self) -> None:
        self._mock_model()
        with pytest.raises(
            CommandLineError,
            match="Input 'invalid' does not exist.",
        ):
            autrainer.cli.inference(model="model", input="invalid", output="")

        OmegaConf.save({}, "input.yaml")

        with pytest.raises(
            CommandLineError,
            match="Input 'input.yaml' is not a directory.",
        ):
            autrainer.cli.inference(model="model", input="input.yaml", output="")

        os.makedirs("input", exist_ok=True)

        with pytest.raises(
            CommandLineError,
            match="No 'example' files found in 'input'.",
        ):
            autrainer.cli.inference(
                model="model",
                input="input",
                output="",
                extension="example",
            )

    def test_invalid_device(
        self,
    ) -> None:
        self._mock_model()
        self._mock_input()
        os.makedirs("model/_best", exist_ok=True)
        OmegaConf.save({}, "model/_best/model.pt")

        with pytest.raises(
            CommandLineError,
            match="Invalid device 'invalid'.",
        ):
            autrainer.cli.inference(
                model="model",
                input="input",
                output="output",
                device="invalid",
            )

    def test_invalid_checkpoint(
        self,
    ) -> None:
        self._mock_model()
        self._mock_input()

        with pytest.raises(
            CommandLineError,
            match="Checkpoint 'invalid' does not exist.",
        ):
            autrainer.cli.inference(
                model="model",
                input="input",
                output="output",
                checkpoint="invalid",
            )

    def test_invalid_preprocess_config(
        self,
    ) -> None:
        self._mock_model()
        self._mock_input()
        os.makedirs("model/_best", exist_ok=True)
        OmegaConf.save({}, "model/_best/model.pt")

        with pytest.raises(
            CommandLineError,
            match="Preprocessing configuration 'invalid.yaml' does not exist.",
        ):
            autrainer.cli.inference(
                model="model",
                input="input",
                output="output",
                preprocess_cfg="invalid",
            )


class BaseCLIRemove(BaseIndividualTempDir):
    def _mock_unsuccessful_run(self, name: str) -> None:
        base = f"results/default/training/{name}"
        os.makedirs(f"{base}/_best", exist_ok=True)
        os.makedirs(f"{base}/epoch_1", exist_ok=True)
        os.makedirs(f"{base}/epoch_2", exist_ok=True)
        OmegaConf.save({}, f"{base}/_best/model.pt")
        OmegaConf.save({}, f"{base}/epoch_1/model.pt")
        OmegaConf.save({}, f"{base}/epoch_2/model.pt")

    def _mock_successful_run(self, name: str) -> None:
        self._mock_unsuccessful_run(name)
        OmegaConf.save({}, f"results/default/training/{name}/metrics.csv")


class TestCLIRmFailed(BaseCLIRemove):
    @pytest.mark.parametrize(
        ("names", "successful"),
        [
            (["run1", "run2", "run3"], [False, False, False]),
            (["run1", "run2", "run3"], [False, True, False]),
            (["run1", "run2", "run3"], [True, True, True]),
        ],
    )
    def test_rm_failed(self, names: List[str], successful: List[bool]) -> None:
        for name, success in zip(names, successful, strict=False):
            if success:
                self._mock_successful_run(name)
            else:
                self._mock_unsuccessful_run(name)
        with patch("builtins.input", return_value="y"):
            autrainer.cli.rm_failed("results", "default")
        for name, success in zip(names, successful, strict=False):
            assert os.path.exists(f"results/default/training/{name}") == success, (
                "Should remove unsuccessful runs."
            )


class TestCLIRmStates(BaseCLIRemove):
    @pytest.mark.parametrize("names", [["run1", "run2", "run3"], []])
    def test_delete_all(self, names: List[str]) -> None:
        for name in names:
            self._mock_successful_run(name)
        with patch("builtins.input", return_value="y"):
            if names:
                autrainer.cli.rm_states("results", "default", keep_best=False)
        for name in names:
            for state in ["_best", "epoch_1", "epoch_2"]:
                assert not os.path.exists(
                    f"results/default/training/{name}/{state}/model.pt"
                ), "Should remove all states."

    @pytest.mark.parametrize("names", [["run1", "run2", "run3"]])
    def test_delete_keep_best(self, names: List[str]) -> None:
        for name in names:
            self._mock_successful_run(name)
        with patch("builtins.input", return_value="y"):
            autrainer.cli.rm_states("results", "default", keep_best=True)
        for name in names:
            for state in ["epoch_1", "epoch_2"]:
                assert not os.path.exists(
                    f"results/default/training/{name}/{state}/model.pt"
                ), "Should remove all states."
        for name in names:
            assert os.path.exists(f"results/default/training/{name}/_best/model.pt"), (
                "Should keep best state."
            )

    @pytest.mark.parametrize("names", [["run1", "run2", "run3"]])
    def test_delete_keep_runs(self, names: List[str]) -> None:
        for name in names:
            self._mock_successful_run(name)
        with patch("builtins.input", return_value="y"):
            autrainer.cli.rm_states(
                "results", "default", keep_runs=[names[0]], keep_best=False
            )
        for name in names[1:]:
            for state in ["_best", "epoch_1", "epoch_2"]:
                assert not os.path.exists(
                    f"results/default/training/{name}/{state}/model.pt"
                ), "Should remove all states."
        for state in ["_best", "epoch_1", "epoch_2"]:
            assert os.path.exists(
                f"results/default/training/{names[0]}/{state}/model.pt"
            ), "Should keep states for specified runs."

    @pytest.mark.parametrize(("names", "keep_it"), [(["run1", "run2", "run3"], 2)])
    def test_delete_keep_iterations(self, names: List[str], keep_it: int) -> None:
        for name in names:
            self._mock_successful_run(name)
        with patch("builtins.input", return_value="y"):
            autrainer.cli.rm_states(
                "results",
                "default",
                keep_iterations=[keep_it],
                keep_best=False,
            )
        for name in names:
            for state in ["_best", "epoch_1"]:
                assert not os.path.exists(
                    f"results/default/training/{name}/{state}/model.pt"
                ), "Should remove all states."
            assert os.path.exists(
                f"results/default/training/{name}/epoch_2/model.pt"
            ), "Should keep states for specified iterations."
        for name in names:
            assert os.path.exists(
                f"results/default/training/{name}/epoch_{keep_it}/model.pt"
            ), "Should keep states for specified iterations."


class TestCLIGroup: ...  # TODO: Add successful grouping test

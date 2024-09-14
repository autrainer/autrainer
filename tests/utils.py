import os
import tempfile
from typing import Generator

import pytest


class BaseSharedTempDir:
    @classmethod
    def setup_class(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.curr_dir = os.getcwd()
        os.chdir(cls.temp_dir.name)

    @classmethod
    def teardown_class(cls) -> None:
        os.chdir(cls.curr_dir)
        cls.temp_dir.cleanup()


class BaseIndividualTempDir:
    previous_temp_dir = None

    @pytest.fixture(autouse=True)
    def isolated_directory(
        self,
        tmp_path: pytest.TempPathFactory,
        request: pytest.FixtureRequest,
    ) -> Generator[None, None, None]:
        if hasattr(request.node, "callspec"):
            persist_previous = request.node.callspec.params.get(
                "reuse_temp_dir", False
            )
        else:
            persist_previous = False

        if persist_previous and BaseIndividualTempDir.previous_temp_dir:
            self.temp_dir = BaseIndividualTempDir.previous_temp_dir
        else:
            self.temp_dir = tmp_path
            BaseIndividualTempDir.previous_temp_dir = tmp_path

        self.curr_dir = os.getcwd()
        os.chdir(self.temp_dir)
        yield
        os.chdir(self.curr_dir)

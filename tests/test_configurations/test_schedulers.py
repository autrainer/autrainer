from typing import Any, Dict

import pytest

from .utils import (
    get_class_from_import_path,
    get_required_parameters,
    load_configurations,
)


class TestSchedulerConfigurations:
    @pytest.mark.parametrize("name, config", load_configurations("scheduler"))
    def test_configurations(self, name: str, config: Dict[str, Any]) -> None:
        assert config.get("id"), f"{name}: Missing ID in configuration"
        if config["id"] == "None":
            return
        assert config.get("_target_"), f"{name}: Missing target"

        cls = get_class_from_import_path(config["_target_"])
        required_params = get_required_parameters(cls, ["optimizer"])
        for arg in required_params:
            assert arg in config, f"{name}: Missing required argument: {arg}"

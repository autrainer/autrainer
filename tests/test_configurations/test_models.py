from typing import Any, Dict

import pytest

from .test_transforms import TestTransformConfigurations
from .utils import (
    get_class_from_import_path,
    get_required_parameters,
    load_configurations,
)


class TestModelConfigurations:
    @pytest.mark.parametrize(("name", "config"), load_configurations("model"))
    def test_configurations(self, name: str, config: Dict[str, Any]) -> None:
        assert config.get("id"), f"{name}: Missing ID in configuration"
        assert config.get("_target_"), f"{name}: Missing target"
        assert config.get("transform"), f"{name}: Missing transform configuration"
        assert config["transform"].get("type"), f"{name}: Missing transform type"
        assert config["transform"]["type"] in [
            "image",
            "grayscale",
            "raw",
            "tabular",
        ], f"{name}: Invalid transform type: {config['transform']['type']}"

        cls = get_class_from_import_path(config["_target_"])
        required_params = get_required_parameters(cls, ["output_dim"])
        for arg in required_params:
            assert arg in config, f"{name}: Missing required argument: {arg}"

        for transform_subset in ["base", "train", "dev", "test"]:
            if config["transform"].get(transform_subset):
                for t in config["transform"][transform_subset]:
                    TestTransformConfigurations._assert_transform(t, name)

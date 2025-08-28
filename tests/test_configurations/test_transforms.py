from typing import Any, Dict, Union

import pytest

from .utils import (
    get_class_from_import_path,
    get_required_parameters,
    load_configurations,
)


AUG_PATH = "autrainer.augmentations.AugmentationPipeline"


class TestTransformConfigurations:
    @pytest.mark.parametrize(
        ("name", "config"),
        load_configurations("preprocessing"),
    )
    def test_preprocesing_configurations(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> None:
        assert config.get("file_handler"), f"{name}: Missing file handler"
        self._assert_transform(config["file_handler"], name)
        assert config.get("pipeline"), f"{name}: Missing pipeline"
        for p in config["pipeline"]:
            self._assert_transform(p, name)

    @pytest.mark.parametrize(
        ("name", "config"),
        load_configurations("augmentation"),
    )
    def test_augmentation_configurations(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> None:
        assert config.get("id"), f"{name}: Missing ID in configuration"
        assert config.get("_target_") == AUG_PATH, f"{name}: Invalid target"
        if config["id"] == "None":
            return
        assert config.get("pipeline"), f"{name}: Missing pipeline"
        for p in config["pipeline"]:
            self._assert_transform(p, name)

    @staticmethod
    def _assert_transform(
        config: Union[str, Dict[str, Any]],
        name: str,
    ) -> None:
        if isinstance(config, str):
            cls = get_class_from_import_path(config)
            values = {}
        else:
            target = next(iter(config.keys()))
            cls = get_class_from_import_path(target)
            values = config[target]
            if not values:  # if the transform is removed by a None value
                return
        required_params = get_required_parameters(cls)
        for arg in required_params:
            assert arg in values, f"{name}: Missing required argument: {arg}"

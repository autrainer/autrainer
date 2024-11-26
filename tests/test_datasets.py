import os
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import torch

from autrainer.datasets import (
    AIBO,
    EDANSA2019,
    AbstractDataset,
    BaseClassificationDataset,
    BaseMLClassificationDataset,
    BaseMTRegressionDataset,
    BaseRegressionDataset,
    DCASE2016Task1,
    DCASE2018Task3,
    DCASE2020Task1A,
    EmoDB,
    SpeechCommands,
    ToyDataset,
)
from autrainer.datasets.utils import AbstractTargetTransform

from .utils import BaseIndividualTempDir


class MockTargetTransform(AbstractTargetTransform):
    def encode(self, x: int) -> int:
        return x

    def decode(self, x: int) -> int:
        return x


class MockBaseInvalidDataset(AbstractDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(task="invalid", **kwargs)

    @property
    def target_transform(self) -> AbstractTargetTransform:
        return MockTargetTransform()

    @property
    def output_dim(self) -> int:
        return 10


class TestBaseDatasets(BaseIndividualTempDir):
    @staticmethod
    def _mock_dataframes(
        path: str,
        index_column: str = "index",
        target_column: str = "target",
        file_type: str = "npy",
        target_type: str = "classification",
        num_files: int = 10,
        output_files: Optional[List[str]] = None,
    ) -> None:
        assert target_type in [
            "classification",
            "ml-classification",
            "regression",
            "mt-regression",
        ]
        os.makedirs(os.path.join(path, "default"), exist_ok=True)
        df = pd.DataFrame()
        df[index_column] = [f"file{i}.{file_type}" for i in range(num_files)]
        if target_type == "classification":
            df[target_column] = [i % 10 for i in range(num_files)]
        elif target_type == "regression":
            df[target_column] = [i for i in range(num_files)]
        elif target_type == "ml-classification":
            for i in range(10):
                df[f"target_{i}"] = torch.randint(0, 2, (num_files,)).tolist()
        else:
            for i in range(10):
                df[f"target_{i}"] = torch.rand(num_files).tolist()

        output_files = output_files or ["train", "dev", "test"]
        for output_file in output_files:
            df.to_csv(os.path.join(path, f"{output_file}.csv"), index=False)

    @staticmethod
    def _mock_data(
        path: str,
        shape: Tuple[int, ...],
        features_subdir: str = "default",
        index_column: str = "index",
        output_files: Optional[List[str]] = None,
    ) -> None:
        if output_files is None:
            output_files = ["train", "dev", "test"]
        dfs = [
            pd.read_csv(os.path.join(path, f"{f}.csv")) for f in output_files
        ]
        for df in dfs:
            for filename in df[index_column]:
                np.save(
                    os.path.join(path, features_subdir, filename),
                    np.random.rand(*shape),
                )

    @staticmethod
    def _mock_dataset_kwargs() -> dict:
        return {
            "path": "data/TestDataset",
            "features_subdir": "default",
            "seed": 42,
            "metrics": ["autrainer.metrics.Accuracy"],
            "tracking_metric": "autrainer.metrics.Accuracy",
            "index_column": "index",
            "target_column": "target",
            "file_type": "npy",
            "file_handler": "autrainer.datasets.utils.NumpyFileHandler",
            "batch_size": 4,
        }

    def test_invalid_task(self) -> None:
        os.makedirs("data/TestDataset/default", exist_ok=True)
        with pytest.raises(ValueError):
            MockBaseInvalidDataset(**self._mock_dataset_kwargs())

    @pytest.mark.parametrize("path", ["data", "data/TestDataset"])
    def test_invalid_path(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with pytest.raises(ValueError):
            BaseClassificationDataset(**self._mock_dataset_kwargs())

    def test_invalid_stratify(self) -> None:
        self._mock_dataframes("data/TestDataset")
        kwargs = self._mock_dataset_kwargs()
        kwargs["stratify"] = ["invalid"]
        with pytest.raises(ValueError):
            BaseClassificationDataset(**kwargs)

    def test_invalid_mlc_stratify(self) -> None:
        self._mock_dataframes(
            "data/TestDataset",
            target_type="ml-classification",
        )
        kwargs = self._mock_dataset_kwargs()
        kwargs["metrics"] = ["autrainer.metrics.MLAccuracy"]
        kwargs["tracking_metric"] = "autrainer.metrics.MLAccuracy"
        kwargs["target_column"] = [f"target_{i}" for i in range(10)]
        kwargs["stratify"] = ["target_1", "target_2"]
        with pytest.raises(ValueError):
            BaseMLClassificationDataset(**kwargs)

    @pytest.mark.parametrize("threshold", [-0.01, 1.1])
    def test_invalid_mlc_threshold(self, threshold: float) -> None:
        self._mock_dataframes(
            "data/TestDataset",
            target_type="ml-classification",
        )
        kwargs = self._mock_dataset_kwargs()
        kwargs["metrics"] = ["autrainer.metrics.MLAccuracy"]
        kwargs["tracking_metric"] = "autrainer.metrics.MLAccuracy"
        kwargs["target_column"] = [f"target_{i}" for i in range(10)]
        kwargs["threshold"] = threshold
        with pytest.raises(ValueError):
            BaseMLClassificationDataset(**kwargs)

    @pytest.mark.parametrize(
        "target_column",
        ["target_1", [], ["target_1", "target_100"]],
    )
    def test_invalid_mlc_target_column(
        self, target_column: Union[str, List[str]]
    ) -> None:
        self._mock_dataframes(
            "data/TestDataset",
            target_type="ml-classification",
        )
        kwargs = self._mock_dataset_kwargs()
        kwargs["metrics"] = ["autrainer.metrics.MLAccuracy"]
        kwargs["tracking_metric"] = "autrainer.metrics.MLAccuracy"
        kwargs["target_column"] = target_column
        with pytest.raises(ValueError):
            BaseMLClassificationDataset(**kwargs)

    def test_classification(self) -> None:
        self._mock_dataframes("data/TestDataset")
        self._mock_data("data/TestDataset", (101, 64))
        data = BaseClassificationDataset(**self._mock_dataset_kwargs())
        self._test_data(data, 10, (4,))

    def test_mlc(self) -> None:
        self._mock_dataframes(
            "data/TestDataset",
            target_type="ml-classification",
        )
        self._mock_data("data/TestDataset", (101, 64))
        kwargs = self._mock_dataset_kwargs()
        kwargs["metrics"] = ["autrainer.metrics.MLAccuracy"]
        kwargs["tracking_metric"] = "autrainer.metrics.MLAccuracy"
        kwargs["target_column"] = [f"target_{i}" for i in range(10)]
        data = BaseMLClassificationDataset(**kwargs)
        self._test_data(data, 10, (4, 10))

    def test_regression(self) -> None:
        self._mock_dataframes("data/TestDataset", target_type="regression")
        self._mock_data("data/TestDataset", (101, 64))
        kwargs = self._mock_dataset_kwargs()
        kwargs["metrics"] = ["autrainer.metrics.MSE"]
        kwargs["tracking_metric"] = "autrainer.metrics.MSE"
        data = BaseRegressionDataset(**kwargs)
        self._test_data(data, 1, (4,))

    def test_mtr(self) -> None:
        self._mock_dataframes("data/TestDataset", target_type="mt-regression")
        self._mock_data("data/TestDataset", (101, 64))
        kwargs = self._mock_dataset_kwargs()
        kwargs["metrics"] = ["autrainer.metrics.MSE"]
        kwargs["tracking_metric"] = "autrainer.metrics.MSE"
        kwargs["target_column"] = [f"target_{i}" for i in range(10)]
        data = BaseMTRegressionDataset(**kwargs)
        self._test_data(data, 10, (4, 10))

    def _test_data(
        self,
        data: AbstractDataset,
        output_dim: str,
        y_shape: Tuple[int, ...],
    ) -> None:
        assert data.output_dim == output_dim, f"Should be {output_dim}."
        assert isinstance(data.df_dev, pd.DataFrame), "Should be a DataFrame."
        assert isinstance(data.df_test, pd.DataFrame), "Should be a DataFrame."
        assert isinstance(
            data.target_transform,
            AbstractTargetTransform,
        ), "Should be an instance of AbstractTargetTransform."

        assert isinstance(
            data.train_dataset, torch.utils.data.Dataset
        ), "Should be an instance of Dataset."
        assert isinstance(
            data.dev_dataset, torch.utils.data.Dataset
        ), "Should be an instance of Dataset."
        assert isinstance(
            data.test_dataset, torch.utils.data.Dataset
        ), "Should be an instance of Dataset."

        assert len(data.train_dataset) == 10, "Should be 10."
        assert len(data.dev_dataset) == 10, "Should be 10."
        assert len(data.test_dataset) == 10, "Should be 10."

        assert isinstance(
            data.train_loader, torch.utils.data.DataLoader
        ), "Should be an instance of DataLoader."
        assert isinstance(
            data.dev_loader, torch.utils.data.DataLoader
        ), "Should be an instance of DataLoader."
        assert isinstance(
            data.test_loader, torch.utils.data.DataLoader
        ), "Should be an instance of DataLoader."

        for loader in [data.train_loader, data.dev_loader, data.test_loader]:
            x = next(iter(loader))
            assert x.features.shape == (4, 101, 64), "Should be (4, 101, 64)."
            assert x.target.shape == y_shape, f"Should be {y_shape}."


class TestAIBO(BaseIndividualTempDir):
    @staticmethod
    def _mock_dataframes(
        path: str,
        index_column: str = "file",
        target_column: str = "target",
        file_type: str = "npy",
        num_files: int = 10,
        task: str = "2cl",
    ) -> None:
        os.makedirs(os.path.join(path, "default"), exist_ok=True)
        df = pd.DataFrame()
        random.seed(42)

        def school() -> str:
            if random.random() < 0.5:
                return "Ohm"
            else:
                return "Mont"

        df["id"] = [f"{school()}_file{i}" for i in range(num_files)]
        df[index_column] = df["id"].apply(lambda x: f"{x}.{file_type}")
        df[target_column] = [i % 10 for i in range(num_files)]

        name = f"chunk_labels_{task}_corpus.txt"
        df.to_csv(os.path.join(path, name), index=False, header=None, sep=" ")

    @staticmethod
    def _mock_data(
        path: str,
        shape: Tuple[int, ...],
        features_subdir: str = "default",
        task: str = "2cl",
    ) -> None:
        name = f"chunk_labels_{task}_corpus.txt"
        df = pd.read_csv(os.path.join(path, name), header=None, sep=" ")
        for filename in df[0]:
            np.save(
                os.path.join(path, features_subdir, filename),
                np.random.rand(*shape),
            )

    @staticmethod
    def _mock_dataset_kwargs() -> dict:
        return {
            "path": "data/TestDataset",
            "features_subdir": "default",
            "seed": 42,
            "metrics": ["autrainer.metrics.Accuracy"],
            "tracking_metric": "autrainer.metrics.Accuracy",
            "index_column": "file",
            "target_column": "class",
            "file_type": "npy",
            "file_handler": "autrainer.datasets.utils.NumpyFileHandler",
            "batch_size": 4,
        }

    def test_invalid_aibo_task(self) -> None:
        with pytest.raises(ValueError):
            AIBO(aibo_task="invalid", **self._mock_dataset_kwargs())

    @pytest.mark.parametrize("aibo_task", ["2cl", "5cl"])
    def test_load_dataframes(self, aibo_task: str) -> None:
        self._mock_dataframes(
            "data/TestDataset",
            task=aibo_task,
        )
        AIBO(**self._mock_dataset_kwargs(), aibo_task=aibo_task)


class TestEDANSA2019(BaseIndividualTempDir):
    def test_invalid_target_column(self) -> None:
        TestBaseDatasets._mock_dataframes(
            "data/TestDataset", target_type="ml-classification"
        )
        kwargs = TestBaseDatasets._mock_dataset_kwargs()
        kwargs["metrics"] = ["autrainer.metrics.MLAccuracy"]
        kwargs["tracking_metric"] = "autrainer.metrics.MLAccuracy"
        kwargs["target_column"] = ["target_1", "target_2"]
        with pytest.raises(ValueError):
            EDANSA2019(**kwargs)


class TestDCASE2016Task1(BaseIndividualTempDir):
    @pytest.mark.parametrize("fold", [-1, 0, 5, 6])
    def test_invalid_fold(self, fold: int) -> None:
        with pytest.raises(ValueError):
            DCASE2016Task1(
                **TestBaseDatasets._mock_dataset_kwargs(),
                fold=fold,
            )

    def test_load_dataframes(self) -> None:
        TestBaseDatasets._mock_dataframes(
            "data/TestDataset",
            output_files=["fold1_train", "fold1_evaluate", "test"],
        )
        DCASE2016Task1(**TestBaseDatasets._mock_dataset_kwargs(), fold=1)


class TestDCASE2018Task3(BaseIndividualTempDir):
    @pytest.mark.parametrize("split", [-0.01, 1.0, 1.1])
    def test_invalid_dev_split(self, split: float) -> None:
        with pytest.raises(ValueError):
            DCASE2018Task3(
                **TestBaseDatasets._mock_dataset_kwargs(),
                dev_split=split,
            )

    def test_load_dataframes(self) -> None:
        TestBaseDatasets._mock_dataframes(
            "data/TestDataset",
            output_files=["train", "test"],
        )
        data = DCASE2018Task3(
            **TestBaseDatasets._mock_dataset_kwargs(),
            dev_split=0.2,
        )
        assert len(data.train_dataset) == 8, "Should be 8."
        assert len(data.dev_dataset) == 2, "Should be 2."
        assert len(data.test_dataset) == 10, "Should be 10."


class TestDCASE2020Task1A(BaseIndividualTempDir):
    targets = [
        "airport",
        "shopping_mall",
        "metro_station",
        "park",
        "bus",
        "metro",
        "tram",
        "airport",
        "shopping_mall",
        "metro_station",
    ]

    def _mock_dcase2020_columns(self, replace_targets: bool = False) -> None:
        for subset in ["train", "test"]:
            df = pd.read_csv(f"data/TestDataset/{subset}.csv")
            df["location"] = [f"loc{i%5}" for i in range(len(df))]
            df["city"] = [f"city{i%3}" for i in range(len(df))]
            df["device"] = [f"device{i%2}" for i in range(len(df))]
            if replace_targets:
                df["target"] = self.targets
            df.to_csv(f"data/TestDataset/{subset}.csv", index=False)

    def test_invalid_scene_category(self) -> None:
        with pytest.raises(ValueError):
            DCASE2020Task1A(
                **TestBaseDatasets._mock_dataset_kwargs(),
                scene_category="invalid",
            )

    def test_dev_split(self) -> None:
        TestBaseDatasets._mock_dataframes(
            "data/TestDataset",
            output_files=["train", "test"],
        )
        self._mock_dcase2020_columns()
        data = DCASE2020Task1A(
            **TestBaseDatasets._mock_dataset_kwargs(),
            dev_split=0.2,
        )
        assert len(data.train_dataset) < 10, "Should be less than 10."
        assert len(data.dev_dataset) > 0, "Should be greater than 0."
        assert len(data.test_dataset) == 10, "Should be 10."

    def test_exclude_cities(self) -> None:
        TestBaseDatasets._mock_dataframes(
            "data/TestDataset",
            output_files=["train", "test"],
        )
        self._mock_dcase2020_columns()
        data = DCASE2020Task1A(
            **TestBaseDatasets._mock_dataset_kwargs(),
            exclude_cities=["city1"],
        )
        assert len(data.train_dataset) == 7, "Should be 7."
        assert len(data.dev_dataset) == 10, "Should be 10."
        assert len(data.test_dataset) == 10, "Should be 10."

    def test_filter_category(self) -> None:
        TestBaseDatasets._mock_dataframes(
            "data/TestDataset",
            output_files=["train", "test"],
        )
        self._mock_dcase2020_columns(replace_targets=True)
        data = DCASE2020Task1A(
            **TestBaseDatasets._mock_dataset_kwargs(),
            scene_category="indoor",
        )
        assert len(data.train_dataset) == 6, "Should be 6."
        assert len(data.dev_dataset) == 6, "Should be 6."
        assert len(data.test_dataset) == 6, "Should be 6."


class TestEmoDB(BaseIndividualTempDir):
    def test_load_dataframes(self) -> None:
        TestBaseDatasets._mock_dataframes(
            "data/TestDataset",
            output_files=["metadata"],
        )
        df = pd.read_csv("data/TestDataset/metadata.csv")
        df["speaker"] = [f"{i%3}" for i in range(len(df))]
        df.to_csv("data/TestDataset/metadata.csv", index=False)
        data = EmoDB(
            **TestBaseDatasets._mock_dataset_kwargs(),
            train_speakers=[0],
            test_speakers=[1],
            dev_speakers=[2],
        )
        assert len(data.train_dataset) == 4, "Should be 4."
        assert len(data.dev_dataset) == 3, "Should be 3."
        assert len(data.test_dataset) == 3, "Should be 3."


class TestSpeechCommands(BaseIndividualTempDir):
    def test_load_dataframes(self) -> None:
        TestBaseDatasets._mock_dataframes(
            "data/TestDataset",
            output_files=["train", "dev", "test"],
        )
        data = SpeechCommands(**TestBaseDatasets._mock_dataset_kwargs())

        assert len(data.train_dataset) == 10, "Should be 10."
        assert len(data.dev_dataset) == 10, "Should be 10."
        assert len(data.test_dataset) == 10, "Should be 10."


class TestToyDataset(BaseIndividualTempDir):
    def _mock_toy_dataset_kwargs(self) -> dict:
        return {
            "task": "classification",
            "size": 100,
            "num_targets": 10,
            "feature_shape": (101, 64),
            "dev_split": 0.2,
            "test_split": 0.2,
            "seed": 42,
            "metrics": ["autrainer.metrics.Accuracy"],
            "tracking_metric": "autrainer.metrics.Accuracy",
            "batch_size": 4,
        }

    def test_invalid_task(self) -> None:
        with pytest.raises(ValueError):
            kwargs = self._mock_toy_dataset_kwargs()
            kwargs["task"] = "invalid"
            ToyDataset(**kwargs)

    @pytest.mark.parametrize(
        "dev_split, test_split",
        [(0.6, 0.5), (0.2, 0.8), (0.8, 0.2)],
    )
    def test_invalid_splits(self, dev_split: float, test_split: float) -> None:
        kwargs = self._mock_toy_dataset_kwargs()
        kwargs["dev_split"] = dev_split
        kwargs["test_split"] = test_split
        with pytest.raises(ValueError):
            ToyDataset(**kwargs)

    def test_invalid_dtype(self) -> None:
        with pytest.raises(ValueError):
            ToyDataset(
                **self._mock_toy_dataset_kwargs(),
                dtype="invalid",
            )

    @pytest.mark.parametrize("size", [-1, 0, 1, 2, 3])
    def test_invalid_size(self, size: int) -> None:
        kwargs = self._mock_toy_dataset_kwargs()
        kwargs["size"] = size
        with pytest.raises(ValueError):
            ToyDataset(**kwargs).df_train

    @staticmethod
    def _test_data_shapes(
        data: AbstractDataset,
        y_shape: Tuple[int, ...],
    ) -> None:
        assert len(data.train_dataset) == 60, "Should be 60."
        assert len(data.dev_dataset) == 20, "Should be 20."
        assert len(data.test_dataset) == 20, "Should be 20."

        x = next(iter(data.train_loader))
        assert x.features.shape == (4, 101, 64), "Should be (4, 101, 64)."
        assert x.target.shape == y_shape, f"Should be {y_shape}."

    @pytest.mark.parametrize("dtype", ["float32", "uint8"])
    def test_dtype(self, dtype: str) -> None:
        kwargs = self._mock_toy_dataset_kwargs()
        data = ToyDataset(**kwargs, dtype=dtype)
        self._test_data_shapes(data, (4,))

    def test_classification(self) -> None:
        data = ToyDataset(**self._mock_toy_dataset_kwargs())
        self._test_data_shapes(data, (4,))
        assert data.output_dim == 10, "Should be 10."

    def test_mlc(self) -> None:
        kwargs = self._mock_toy_dataset_kwargs()
        kwargs["task"] = "ml-classification"
        kwargs["metrics"] = ["autrainer.metrics.MLAccuracy"]
        kwargs["tracking_metric"] = "autrainer.metrics.MLAccuracy"
        data = ToyDataset(**kwargs)
        self._test_data_shapes(data, (4, 10))
        assert data.output_dim == 10, "Should be 10."

    def test_regression(self) -> None:
        kwargs = self._mock_toy_dataset_kwargs()
        kwargs["task"] = "regression"
        kwargs["metrics"] = ["autrainer.metrics.MSE"]
        kwargs["tracking_metric"] = "autrainer.metrics.MSE"
        data = ToyDataset(**kwargs)
        self._test_data_shapes(data, (4,))
        assert data.output_dim == 1, "Should be 1."

    def test_mtr(self) -> None:
        kwargs = self._mock_toy_dataset_kwargs()
        kwargs["task"] = "mt-regression"
        kwargs["metrics"] = ["autrainer.metrics.MSE"]
        kwargs["tracking_metric"] = "autrainer.metrics.MSE"
        data = ToyDataset(**kwargs)
        self._test_data_shapes(data, (4, 10))
        assert data.output_dim == 10, "Should be 10."

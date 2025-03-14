import os
from typing import Dict, List, Optional

import audiofile
import numpy as np
import pandas as pd
import pytest
import torch

from autrainer.core.scripts.preprocess_script import preprocess_main
from autrainer.datasets import BaseClassificationDataset

from .utils import BaseIndividualTempDir


class TestPreprocessing(BaseIndividualTempDir):
    @staticmethod
    def _mock_dataframes(
        path: str,
        index_column: str = "index",
        target_column: str = "target",
        file_type: str = "wav",
        target_type: str = "classification",
        num_files: int = 10,
        output_files: Optional[List[str]] = None,
    ) -> None:
        assert target_type in [
            "classification",
            "ml-classification",
            "regression",
        ]
        os.makedirs(os.path.join(path, "default"), exist_ok=True)
        nested_prefix = {0: "", 1: "foo/", 2: "foo/bar/"}
        df = pd.DataFrame()
        df[index_column] = [
            os.path.join(nested_prefix[i % 3], f"file{i}.{file_type}")
            for i in range(num_files)
        ]
        if target_type == "classification":
            df[target_column] = [i % 10 for i in range(num_files)]
        elif target_type == "regression":
            df[target_column] = [i for i in range(num_files)]
        elif target_type in ["ml-classification", "mt-regression"]:
            for i in range(10):
                df[f"target_{i}"] = torch.randint(0, 2, (num_files,)).tolist()
        else:
            msg = f"Target type '{target_type}' not implemented."
            raise NotImplementedError(msg)

        output_files = output_files or ["train", "dev", "test"]
        for output_file in output_files:
            df.to_csv(os.path.join(path, f"{output_file}.csv"), index=False)

    @staticmethod
    def _mock_data(
        path: str,
        sampling_rate: int,
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
                filepath = os.path.join(path, features_subdir, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                audiofile.write(
                    filepath,
                    np.random.rand(sampling_rate),
                    sampling_rate,
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
            "file_type": "wav",
            "file_handler": "autrainer.datasets.utils.AudioFileHandler",
            "criterion": "autrainer.criterion.CrossEntropyLoss",
            "transform": "wav",
            "_target_": "autrainer.datasets.BaseClassificationDataset",
        }

    @pytest.mark.parametrize(
        "preprocess,sampling_rate,features_path",
        [
            (
                {
                    "file_handler": {
                        "autrainer.datasets.utils.AudioFileHandler": {
                            "target_sample_rate": 16000
                        }
                    },
                    "pipeline": [
                        "autrainer.transforms.StereoToMono",
                        {
                            "autrainer.transforms.PannMel": {
                                "sample_rate": 16000,
                                "window_size": 512,
                                "hop_size": 160,
                                "mel_bins": 64,
                                "fmin": 50,
                                "fmax": 8000,
                                "ref": 1.0,
                                "amin": 1e-10,
                                "top_db": None,
                            }
                        },
                    ],
                },
                16000,
                None,
            ),
            (
                {
                    "file_handler": {
                        "autrainer.datasets.utils.AudioFileHandler": {
                            "target_sample_rate": 16000
                        }
                    },
                    "pipeline": [
                        "autrainer.transforms.StereoToMono",
                        {
                            "autrainer.transforms.PannMel": {
                                "sample_rate": 16000,
                                "window_size": 512,
                                "hop_size": 160,
                                "mel_bins": 64,
                                "fmin": 50,
                                "fmax": 8000,
                                "ref": 1.0,
                                "amin": 1e-10,
                                "top_db": None,
                            }
                        },
                    ],
                },
                16000,
                "foo",
            ),
        ],
    )
    def test_log_mel_spectrogram_preprocessing(
        self,
        preprocess: Dict,
        sampling_rate: int,
        features_path: Optional[str],
    ) -> None:
        self._mock_dataframes("data/TestDataset")
        self._mock_data("data/TestDataset", sampling_rate)

        dataset_args = self._mock_dataset_kwargs()
        criterion = dataset_args.pop("criterion")
        transform = dataset_args.pop("transform")
        target = dataset_args.pop("_target_")
        data = BaseClassificationDataset(**dataset_args)
        for d in (data.train_dataset, data.dev_dataset, data.test_dataset):
            for x in d:
                assert x[0].shape == (
                    1,
                    sampling_rate,
                ), "Should match shape of mono audio."

        df_train_audio = data.df_train
        df_dev_audio = data.df_dev
        df_test_audio = data.df_test

        dataset_args["criterion"] = criterion
        dataset_args["transform"] = transform
        dataset_args["features_path"] = features_path
        dataset_args["_target_"] = target
        dataset_args["features_subdir"] = "log_mel_16k"
        dataset_args["file_type"] = "npy"
        dataset_args["file_handler"] = (
            "autrainer.datasets.utils.NumpyFileHandler"
        )

        preprocess_main(
            name="TestDataset",
            dataset=dataset_args,
            preprocess=preprocess,
            num_workers=1,
            update_frequency=1,
        )

        dataset_args.pop("_target_")
        dataset_args.pop("_convert_")
        dataset_args.pop("_recursive_")
        dataset_args["features_path"] = features_path
        dataset_args["file_type"] = "npy"
        dataset_args["file_handler"] = (
            "autrainer.datasets.utils.NumpyFileHandler"
        )
        dataset_args["features_subdir"] = "log_mel_16k"

        _mel = preprocess["pipeline"][-1]["autrainer.transforms.PannMel"]
        shape_time = sampling_rate // _mel["hop_size"] + 1
        shape_freq = _mel["mel_bins"]

        data = BaseClassificationDataset(**dataset_args)
        for d in (data.train_dataset, data.dev_dataset, data.test_dataset):
            for x in d:
                assert x[0].shape == (
                    1,
                    shape_time,
                    shape_freq,
                ), "Should match shape of log mel spectrogram."

        for df_audio, df_numpy in zip(
            (df_train_audio, df_dev_audio, df_test_audio),
            (data.df_train, data.df_dev, data.df_test),
        ):
            for audio_file, numpy_file in zip(
                df_audio[data.index_column], df_numpy[data.index_column]
            ):
                assert audio_file == numpy_file.replace(
                    "npy", "wav"
                ), "Should match audio file path."

import glob
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import audobject
from omegaconf import OmegaConf
import pandas as pd
import torch
from tqdm import tqdm
import yaml

import autrainer
from autrainer.core.utils import set_device
from autrainer.datasets.utils import (
    AbstractFileHandler,
    AbstractTargetTransform,
)
from autrainer.models import AbstractModel
from autrainer.transforms import SmartCompose


class Inference:
    def __init__(
        self,
        model_path: str,
        checkpoint: str = "_best",
        device: str = "cpu",
        preprocess_cfg: Optional[str] = "default",
        window_length: Optional[float] = None,
        stride_length: Optional[float] = None,
        min_length: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Inference class for audio models.

        Args:
            model_path: Local model directory containing the model.yaml,
                file_handler.yaml, target_transform.yaml, and
                inference_transform.yaml files.
            checkpoint: Checkpoint directory containing a model.pt file.
                Defaults to "_best".
            device: Device to run inference on. Defaults to "cpu".
            preprocess_cfg: Preprocessing configuration file. If "default",
                the default preprocessing pipeline used during training is
                applied. If None, no preprocessing is applied. Defaults to
                "default".
            window_length: Window length in seconds for sliding window
                inference. Defaults to None.
            stride_length: Stride length in seconds for sliding window
                inference. Defaults to None.
            min_length: Minimum length of an audio file in seconds.
                Audio files shorter than this will be padded with zeros.
                Defaults to None.
            sample_rate: Sample rate of the audio files in Hz.
                Defaults to None.
        """
        self._model_path = model_path
        self._checkpoint = checkpoint
        self._device = set_device(device)
        self._preprocess_cfg = preprocess_cfg
        self._window_length = window_length
        self._stride_length = stride_length
        self._min_length = min_length
        self._sample_rate = sample_rate
        sys.path.append(os.getcwd())

        self.model: AbstractModel = audobject.from_yaml(
            os.path.join(self._model_path, "model.yaml")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            state = torch.load(
                os.path.join(self._model_path, self._checkpoint, "model.pt"),
                map_location="cpu",
                weights_only=True,
            )
        self.model.load_state_dict(state)
        self.model.to(self._device)
        self.model.eval()
        self.file_handler: AbstractFileHandler = audobject.from_yaml(
            os.path.join(self._model_path, "file_handler.yaml")
        )
        self.target_transform: AbstractTargetTransform = audobject.from_yaml(
            os.path.join(self._model_path, "target_transform.yaml")
        )
        self.inference_transform: SmartCompose = audobject.from_yaml(
            os.path.join(self._model_path, "inference_transform.yaml")
        )

        self.preprocess_pipeline = SmartCompose([])
        if self._preprocess_cfg == "default":
            self.file_handler: AbstractFileHandler = audobject.from_yaml(
                os.path.join(self._model_path, "preprocess_file_handler.yaml")
            )
            self.preprocess_pipeline: SmartCompose = audobject.from_yaml(
                os.path.join(self._model_path, "preprocess_pipeline.yaml")
            )
        elif self._preprocess_cfg is not None:
            preprocess_cfg = OmegaConf.to_container(
                OmegaConf.load(self._preprocess_cfg)
            )
            self.file_handler = autrainer.instantiate_shorthand(
                preprocess_cfg["file_handler"]
            )
            pipeline = [
                autrainer.instantiate_shorthand(p)
                for p in preprocess_cfg["pipeline"]
            ]
            self.preprocess_pipeline = SmartCompose(pipeline)

    def predict_directory(
        self,
        directory: str,
        extension: str,
        recursive: bool = False,
        update_frequency: int = 1,
    ) -> pd.DataFrame:
        """Obtain the model predictions for all files in a directory.

        Args:
            directory: Path to the directory containing audio files.
            extension: File extension of the audio files.
            recursive: Whether to search recursively for audio files in
                subdirectories. Defaults to False.
            update_frequency: Frequency of progress bar updates. If 0, the
                progress bar will be disabled. Defaults to 1.

        Returns:
            DataFrame containing the filename, prediction, and output for
            each file. If sliding window inference is used, the offset is
            additionally included.
        """
        files = self._collect_files(directory, extension, recursive)
        records = []
        for file in tqdm(
            files,
            disable=update_frequency == 0,
            miniters=update_frequency,
            desc="Inference prediction",
        ):
            prediction, output, probs = self.predict_file(file)
            if isinstance(prediction, dict):
                for (offset, pred), out in zip(
                    prediction.items(), output.values()
                ):
                    records.append(
                        {
                            "filename": os.path.relpath(file, directory),
                            "offset": offset,
                            "prediction": pred,
                            "output": out.squeeze().tolist(),
                            **probs[offset],
                        }
                    )
            else:
                records.append(
                    {
                        "filename": os.path.relpath(file, directory),
                        "prediction": prediction,
                        "output": output.squeeze().tolist(),
                        **probs,
                    }
                )
        return pd.DataFrame(records)

    def embed_directory(
        self,
        directory: str,
        extension: str,
        recursive: bool = False,
        update_frequency: int = 1,
    ) -> pd.DataFrame:
        """Obtain the model embeddings for all files in a directory.

        Args:
            directory: Path to the directory containing audio files.
            extension: File extension of the audio files.
            recursive: Whether to search recursively for audio files in
                subdirectories. Defaults to False.
            update_frequency: Frequency of progress bar updates. If 0, the
            progress bar will be disabled. Defaults to 1.

        Returns:
            DataFrame containing the filename and embedding for each file.
            If sliding window inference is used, the offset is additionally
            included.
        """
        files = self._collect_files(directory, extension, recursive)
        records = []
        for file in tqdm(
            files,
            disable=update_frequency == 0,
            miniters=update_frequency,
            desc="Inference embedding",
        ):
            embedding = self.embed_file(file)
            if isinstance(embedding, dict):
                for offset, emb in embedding.items():
                    records.append(
                        {
                            "filename": os.path.relpath(file, directory),
                            "offset": offset,
                            "embedding": emb,
                        }
                    )
            else:
                records.append(
                    {
                        "filename": os.path.relpath(file, directory),
                        "embedding": embedding,
                    }
                )
        return pd.DataFrame(records)

    def predict_file(
        self,
        file: str,
    ) -> Union[
        Tuple[Any, torch.Tensor, Dict[str, float]],
        Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[str, dict]],
    ]:
        """Obtain the model prediction for a single file.

        Args:
            file: Path to the audio file.

        Returns:
            Model prediction, output, and probabilties for the file.
            If sliding window inference is used, the prediction is a dictionary
            with the offset as the key.
        """
        return self._delegate_file(file, self._predict, self._predict_windowed)

    def embed_file(
        self,
        file: str,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Obtain the model embedding for a single file.

        Args:
            file: Path to the audio file.

        Returns:
            Model embedding for the file. If sliding window inference is used,
            the embedding is a dictionary with the offset as the key.
        """
        return self._delegate_file(file, self._embed, self._embed_windowed)

    @staticmethod
    def save_prediction_yaml(results: pd.DataFrame, output_dir: str) -> None:
        """Save the prediction results to a YAML file.

        Creates a human-readable YAML file with the prediction results.

        Args:
            results: DataFrame containing the results.
            output_dir: Output directory to save the results to.
        """
        os.makedirs(output_dir, exist_ok=True)
        if "offset" in results.columns:
            results = (
                results.set_index(["filename", "offset"])["prediction"]
                .unstack(0)
                .to_dict()
            )
        else:
            results = results.set_index("filename")["prediction"].to_dict()
        with open(os.path.join(output_dir, "results.yaml"), "w") as fp:
            yaml.dump(results, fp)

    @staticmethod
    def save_prediction_results(
        results: pd.DataFrame,
        output_dir: str,
    ) -> None:
        """Save the prediction results to a CSV file.

        Creates a CSV file with the model predictions and outputs.

        Args:
            results: DataFrame containing the results.
            output_dir: Output directory to save the results to.
        """
        os.makedirs(output_dir, exist_ok=True)
        results.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    @staticmethod
    def save_embeddings(
        results: pd.DataFrame,
        output_dir: str,
        input_extension: str,
    ) -> None:
        """Save the embeddings as torch tensors.

        Saves the embeddings to the output directory with the same filename
        as each audio file.

        Args:
            results: DataFrame containing the embeddings.
            output_dir: Output directory to save the embeddings to.
            input_extension: File extension of the input audio files to replace
                with ".pt".
        """
        os.makedirs(output_dir, exist_ok=True)

        def _create_output_name(row, offset: bool = False) -> str:
            n = row["filename"].replace("." + input_extension, "")
            n += f"-{row['offset']}" if offset else ""
            return n + ".pt"

        for _, row in results.iterrows():
            output_path = os.path.join(
                output_dir,
                _create_output_name(row, "offset" in results.columns),
            )
            torch.save(row["embedding"], output_path)

    def _delegate_file(
        self,
        file: str,
        fn: Callable,
        window_fn: Callable,
    ) -> Union[Tuple[Any, torch.Tensor], torch.Tensor]:
        x = self.file_handler.load(file)
        if self._window_length and self._stride_length and self._sample_rate:
            return window_fn(x)
        else:
            return fn(x)

    def _collect_files(self, directory: str, extension: str, recursive: bool):
        pattern = f"**/*.{extension}" if recursive else f"*.{extension}"
        return glob.glob(os.path.join(directory, pattern), recursive=True)

    def _preprocess_file(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_audio(x)
        x = self.preprocess_pipeline(x, 0)
        x = self.inference_transform(x, 0)
        x = x.unsqueeze(0).to(self._device)
        return x

    def _predict(
        self,
        x: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor, Dict[str, float]]:
        x = self._preprocess_file(x)
        with torch.inference_mode():
            output = self.model(x).cpu()
        probabilities = self.target_transform.probabilities_inference(output)
        prediction = self.target_transform.predict_inference(probabilities)
        decoded_prediction = self.target_transform.decode(prediction)
        probs_dict = self.target_transform.probabilities_to_dict(
            probabilities.squeeze()
        )
        return decoded_prediction, output, probs_dict

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_file(x)
        with torch.inference_mode():
            embedding = self.model.embeddings(x).cpu().squeeze()
        return embedding

    def _create_windows(self, x: torch.Tensor) -> Tuple[int, int, List[int]]:
        w_len = int(self._window_length * self._sample_rate)
        s_len = int(self._stride_length * self._sample_rate)
        num_windows = (x.shape[1] - w_len) // s_len + 1
        return w_len, s_len, num_windows

    def _predict_windowed(
        self,
        x: torch.Tensor,
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[str, dict]]:
        results = {}
        outputs = {}
        probs = {}
        w_len, s_len, num_windows = self._create_windows(x)
        for i in range(num_windows):
            start_idx = i * s_len
            end_idx = min(start_idx + w_len, x.shape[1])
            prediction, output, prob = self._predict(x[:, start_idx:end_idx])
            start_time = start_idx / self._sample_rate
            end_time = end_idx / self._sample_rate
            results[f"{start_time:.2f}-{end_time:.2f}"] = prediction
            outputs[f"{start_time:.2f}-{end_time:.2f}"] = output
            probs[f"{start_time:.2f}-{end_time:.2f}"] = prob
        majority = self.target_transform.majority_vote(list(results.values()))
        results["majority"] = majority
        outputs["majority"] = torch.empty(0)
        probs["majority"] = {k: None for k in prob.keys()}
        return results, outputs, probs

    def _embed_windowed(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = {}
        w_len, s_len, num_windows = self._create_windows(x)
        for i in range(num_windows):
            start_idx = i * s_len
            end_idx = min(start_idx + w_len, x.shape[1])
            embedding = self._embed(x[:, start_idx:end_idx])
            start_time = start_idx / self._sample_rate
            end_time = end_idx / self._sample_rate
            results[f"{start_time:.2f}-{end_time:.2f}"] = embedding
        return results

    def _pad_audio(self, x: torch.Tensor) -> torch.Tensor:
        if self._min_length is None or self._sample_rate is None:
            return x
        min_length = int(self._min_length * self._sample_rate)
        if x.shape[1] >= min_length:
            return x
        return torch.nn.functional.pad(
            x,
            (0, min_length - x.shape[1]),
            value=0,
        )

    def __repr__(self) -> str:
        s = (
            f"File handler:\n{self.file_handler}"
            f"Data transform:\n{self.inference_transform}"
            f"Model:\n{self.model}"
            f"Label transform:\n{self.target_transform}"
        )
        return s

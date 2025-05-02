import os
import shutil
import tempfile

from omegaconf import OmegaConf
import torch

from autrainer.core.structs import AbstractDataItem
from autrainer.core.utils import Bookkeeping
from autrainer.datasets.utils import LabelEncoder, NumpyFileHandler
from autrainer.models import FFNN
from autrainer.serving import Inference
from autrainer.transforms import AbstractTransform, SmartCompose

from .utils import BaseIndividualTempDir


class MockSqueezeFirstDim(AbstractTransform):
    def __init__(self):
        super().__init__(0)

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        item.features = item.features.squeeze(0)
        return item


class TestInference(BaseIndividualTempDir):
    @classmethod
    def setup_class(cls) -> None:
        cls.base_dir = tempfile.TemporaryDirectory()
        model = FFNN(output_dim=10, input_size=64, hidden_size=64)
        model.eval()
        file_handler = NumpyFileHandler()
        target_t = LabelEncoder([f"target_{i}" for i in range(10)])
        inference_t = SmartCompose([MockSqueezeFirstDim()])
        bookkeeping = Bookkeeping(os.path.join(cls.base_dir.name, "TestModel"))
        bookkeeping.save_audobject(model, "model.yaml")
        bookkeeping.save_state(model, "model.pt", "_best")
        bookkeeping.save_audobject(file_handler, "file_handler.yaml")
        bookkeeping.save_audobject(target_t, "target_transform.yaml")
        bookkeeping.save_audobject(inference_t, "inference_transform.yaml")
        bookkeeping.save_audobject(
            file_handler, "preprocess_file_handler.yaml"
        )
        bookkeeping.save_audobject(inference_t, "preprocess_pipeline.yaml")

    @classmethod
    def teardown_class(cls) -> None:
        cls.base_dir.cleanup()

    def _mock_model_setup(self) -> None:
        shutil.copytree(
            os.path.join(self.base_dir.name, "TestModel"),
            "TestModel",
        )

    def _mock_data_setup(self) -> None:
        os.makedirs("input", exist_ok=True)
        for i in range(10):
            data = torch.randn(1, 64)
            NumpyFileHandler().save(f"input/audio_{i}.npy", data)

    def test_repr(self) -> None:
        self._mock_model_setup()
        str(Inference("TestModel"))

    def test_predict(self) -> None:
        self._mock_model_setup()
        self._mock_data_setup()
        inference = Inference("TestModel")
        df = inference.predict_directory("input", "npy")
        assert len(df) == 10, "Should predict 10 samples."
        assert all(
            c in df.columns for c in ["filename", "prediction", "output"]
        ), "Should have columns: audio, target, prediction."
        inference.save_prediction_results(df, "output")
        inference.save_prediction_yaml(df, "output")
        assert os.path.exists(
            "output/results.csv"
        ), "Should save prediction results."
        assert os.path.exists(
            "output/results.yaml"
        ), "Should save prediction YAML."

        pred, out, probs = inference.predict_file("input/audio_0.npy")
        assert isinstance(pred, str), "Should return a prediction string."
        assert isinstance(out, torch.Tensor), "Should return an output tensor."
        assert isinstance(probs, dict), "Should return a probabilities dict."

    def test_predict_sliding_window(self) -> None:
        self._mock_model_setup()
        self._mock_data_setup()
        inference = Inference(
            model_path="TestModel",
            window_length=2,
            stride_length=1,
            min_length=4,
            sample_rate=16,
        )
        df = inference.predict_directory("input", "npy")
        assert (
            len(df) == 40
        ), "Should predict 10 samples with 3 windows each and majority voting."
        assert all(
            c in df.columns
            for c in ["filename", "offset", "prediction", "output"]
        ), "Should have columns: audio, target, prediction."
        inference.save_prediction_results(df, "output")
        inference.save_prediction_yaml(df, "output")
        assert os.path.exists(
            "output/results.csv"
        ), "Should save prediction results."
        assert os.path.exists(
            "output/results.yaml"
        ), "Should save prediction YAML."

        pred, out, probs = inference.predict_file("input/audio_0.npy")
        assert isinstance(pred, dict), "Should return a prediction dictionary"
        assert isinstance(out, dict), "Should return an output dictionary"
        assert isinstance(probs, dict), "Should return a probabilities dict."

    def test_embed(self) -> None:
        self._mock_model_setup()
        self._mock_data_setup()
        inference = Inference("TestModel")
        df = inference.embed_directory("input", "npy")
        assert len(df) == 10, "Should embed 10 samples."
        assert all(
            c in df.columns for c in ["filename", "embedding"]
        ), "Should have columns: audio, embedding."
        inference.save_embeddings(df, "output", "npy")
        assert all(
            os.path.isfile(f"output/audio_{i}.pt") for i in range(10)
        ), "Should save embeddings."
        emb = inference.embed_file("input/audio_0.npy")
        assert isinstance(
            emb, torch.Tensor
        ), "Should return an embedding tensor."

    def test_embed_sliding_window(self) -> None:
        self._mock_model_setup()
        self._mock_data_setup()
        inference = Inference(
            model_path="TestModel",
            window_length=2,
            stride_length=1,
            min_length=4,
            sample_rate=16,
        )
        df = inference.embed_directory("input", "npy")
        assert len(df) == 30, "Should embed 10 samples with 3 windows each."
        assert all(
            c in df.columns for c in ["filename", "offset", "embedding"]
        ), "Should have columns: audio, embedding."
        inference.save_embeddings(df, "output", "npy")
        audio_embeddings = [
            f for f in os.listdir("output") if f.endswith(".pt")
        ]
        assert len(audio_embeddings) == 30, "Should save embeddings."
        emb = inference.embed_file("input/audio_0.npy")
        assert isinstance(
            emb, dict
        ), "Should return a dictionary of embeddings."

    def test_preprocessing(self) -> None:
        self._mock_model_setup()
        self._mock_data_setup()
        cfg = {
            "file_handler": "autrainer.datasets.utils.NumpyFileHandler",
            "pipeline": [
                {"autrainer.transforms.ScaleRange": {"range": [0, 1]}}
            ],
        }
        OmegaConf.save(cfg, "preprocessing.yaml")
        inference = Inference("TestModel", preprocess_cfg="preprocessing.yaml")
        df = inference.predict_directory("input", "npy")
        assert len(df) == 10, "Should predict 10 samples."
        assert all(
            c in df.columns for c in ["filename", "prediction", "output"]
        ), "Should have columns: audio, target, prediction."
        inference.save_prediction_results(df, "output")
        inference.save_prediction_yaml(df, "output")
        assert os.path.exists(
            "output/results.csv"
        ), "Should save prediction results."
        assert os.path.exists(
            "output/results.yaml"
        ), "Should save prediction YAML."

        pred, out, probs = inference.predict_file("input/audio_0.npy")
        assert isinstance(pred, str), "Should return a prediction string."
        assert isinstance(out, torch.Tensor), "Should return an output tensor."
        assert isinstance(probs, dict), "Should return a probabilities dict."

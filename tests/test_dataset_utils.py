import os
from typing import List, Union

import pytest
import torch

from autrainer.datasets.utils import (
    AudioFileHandler,
    IdentityFileHandler,
    ImageFileHandler,
    LabelEncoder,
    MinMaxScaler,
    MultiLabelEncoder,
    NumpyFileHandler,
)

from .utils import BaseIndividualTempDir


class TestAudioFileHandler(BaseIndividualTempDir):
    def test_invalid_save(self) -> None:
        handler = AudioFileHandler()
        with pytest.raises(ValueError):
            handler.save("audio.wav", torch.randn(1, 16000))

    def test_file_handler(self) -> None:
        handler = AudioFileHandler(target_sample_rate=16000)
        data = torch.randn(1, 16000)
        handler.save("audio.wav", data)
        assert os.path.exists("audio.wav"), "Should save the audio file."
        loaded_data = handler("audio.wav")
        assert data.shape == loaded_data.shape, "Should load the audio file."

    def test_load_differing_sample_rate(self) -> None:
        handler = AudioFileHandler(target_sample_rate=16000)
        data = torch.randn(1, 16000)
        handler.save("audio.wav", data)
        handler = AudioFileHandler(target_sample_rate=8000)
        loaded_data = handler("audio.wav")
        assert (
            data.shape[1] // 2 == loaded_data.shape[1]
        ), "Should load the audio file with the correct sample rate."


class TestAllFileHandlers(BaseIndividualTempDir):
    def test_identity_file_handler(self) -> None:
        handler = IdentityFileHandler()
        assert (
            handler("file") == "file"
        ), "Should serve as an identity function."
        handler.save("file", torch.rand(1))
        assert not os.path.exists("file"), "Should not save anything."

    def test_image_file_handler(self) -> None:
        handler = ImageFileHandler()
        data = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        handler.save("image.png", data)
        assert os.path.exists("image.png"), "Should save the image file."
        loaded_data = handler("image.png")
        assert torch.allclose(data, loaded_data), "Should load the image file."

    def test_numpy_file_handler(self) -> None:
        handler = NumpyFileHandler()
        data = torch.randn(1, 101, 64)
        handler.save("data.npy", data)
        assert os.path.exists("data.npy"), "Should save the numpy file."
        loaded_data = handler("data.npy")
        assert torch.allclose(data, loaded_data), "Should load the numpy file."


def _all_close_dict(a: dict, b: dict) -> bool:
    for key in a.keys():
        if not torch.allclose(torch.tensor([a[key]]), torch.tensor([b[key]])):
            return False
    return True


class TestMinMaxScaler:
    def test_len(self) -> None:
        scaler = MinMaxScaler("target", 0, 1)
        assert len(scaler) == 1, "Should have a single target."

    @pytest.mark.parametrize("minimum, maximum", [(1, 0), (0, 0), (1, 1)])
    def test_invalid_min_max(self, minimum: float, maximum: float) -> None:
        with pytest.raises(ValueError):
            MinMaxScaler("target", minimum, maximum)

    @pytest.mark.parametrize("x", [0, 1, 0.5, 10, -1, -0.5, -10])
    def test_encode_decode(self, x: float) -> None:
        scaler = MinMaxScaler("target", 0, 1)
        assert scaler.decode(scaler(x)) == x, "Should encode and decode."

    def test_probabilities_training(self) -> None:
        scaler = MinMaxScaler("target", 0, 1)
        x = torch.Tensor([[0.1], [0.9], [0.6], [0.4], [0.5]])
        probs = scaler.probabilities_training(x)
        assert torch.all(probs >= 0) and torch.all(
            probs <= 1
        ), "Should be in [0, 1]."

    def test_probabilities_predict(self) -> None:
        scaler = MinMaxScaler("target", 0, 1)
        x = torch.Tensor([[0.1], [0.9], [0.6], [0.4], [0.5]])
        probs = scaler.probabilities_inference(x)
        preds = scaler.predict_inference(probs)
        assert (
            preds == torch.sigmoid(x).squeeze().tolist()
        ), "Should predict the batch."

    def test_majority_vote(self) -> None:
        encoder = MinMaxScaler("target", 0, 1)
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert (
            encoder.majority_vote(x) == 0.3
        ), "Should compute the majority vote."

    def test_probabilities_to_dict(self) -> None:
        scaler = MinMaxScaler("target", 0, 1)
        x = torch.Tensor([0.5])
        probs_dict = scaler.probabilities_to_dict(x)
        assert _all_close_dict(
            probs_dict, {"target": 0.5}
        ), "Should convert the probabilities to a dictionary."


class TestMultiLabelEncoder:
    labels = ["fizz", "buzz", "jazz"]

    def test_len(self) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        assert len(encoder) == 3, "Should have three labels."

    @pytest.mark.parametrize("threshold", [-1, -0.01, 1.1, 2])
    def test_invalid_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError):
            MultiLabelEncoder(threshold, self.labels)

    @pytest.mark.parametrize("labels", [[1, 0, 1], ["fizz", "jazz"]])
    def test_encode_decode(self, labels: Union[List[int], List[str]]) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        assert encoder.decode(encoder(labels).tolist()) == [
            "fizz",
            "jazz",
        ], "Should encode and decode."

    def test_empty_encode(self) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        assert torch.all(
            encoder([]) == torch.zeros(3)
        ), "Should encode an empty list."

    def test_empty_decode(self) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        assert (
            encoder.decode(encoder([]).tolist()) == []
        ), "Should decode an empty list."

    def test_probabilities_training(self) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        x = torch.Tensor([[0.1, 0.9, 0.6], [0.9, 0.1, 0.6]])
        probs = encoder.probabilities_training(x)
        assert torch.allclose(probs, x), "Should be a no-op."

    def test_probabilities_predict(self) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        x = torch.Tensor([-0.1, 0.9, 0.6])
        probs = encoder.probabilities_inference(x)
        preds = encoder.predict_inference(probs)
        assert preds == [0, 1, 1], "Should predict the batch."

    def test_majority_vote(self) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        x = [["fizz", "jazz"], ["fizz"], ["fizz", "buzz"]]
        assert encoder.majority_vote(x) == [
            "fizz"
        ], "Should compute the majority vote."

    def test_probabilities_to_dict(self) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        x = torch.Tensor([0.5, 0.6, 0.7])
        probs = encoder.probabilities_to_dict(x)
        assert _all_close_dict(
            probs, {"fizz": 0.5, "buzz": 0.6, "jazz": 0.7}
        ), "Should convert the probabilities to a dictionary."


class TestLabelEncoder:
    labels = ["fizz", "buzz", "jazz"]

    def test_len(self) -> None:
        encoder = LabelEncoder(self.labels)
        assert len(encoder) == 3, "Should have three labels."

    @pytest.mark.parametrize("label", ["jazz", "buzz", "fizz"])
    def test_encode_decode(self, label: str) -> None:
        encoder = LabelEncoder(self.labels)
        assert (
            encoder.decode(encoder(label)) == label
        ), "Should encode and decode."

    def test_probabilities_training(self) -> None:
        encoder = LabelEncoder(self.labels)
        x = torch.Tensor([0.5, 0.6, 0.7])
        probs = encoder.probabilities_training(x)
        assert torch.allclose(probs, x), "Should be a no-op."

    def test_probabilities_predict(self) -> None:
        encoder = LabelEncoder(self.labels)
        x = torch.Tensor([[0, 1, 0], [0, 0, 1]])
        probs = encoder.probabilities_inference(x)
        preds = encoder.predict_inference(probs)
        assert preds == [1, 2], "Should predict the batch."

    def test_majority_vote(self) -> None:
        encoder = LabelEncoder(self.labels)
        x = ["fizz", "buzz", "jazz", "fizz", "buzz", "fizz"]
        assert (
            encoder.majority_vote(x) == "fizz"
        ), "Should compute the majority vote."

    def test_probabilities_to_dict(self) -> None:
        encoder = LabelEncoder(self.labels)
        x = torch.Tensor([0.5, 0.6, 0.7])
        probs = encoder.probabilities_to_dict(x)
        assert _all_close_dict(
            probs,
            {
                "buzz": 0.5,
                "fizz": 0.6,
                "jazz": 0.7,
            },  # alphabetical order is important
        ), "Should convert the probabilities to a dictionary."

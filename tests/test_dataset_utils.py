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


class TestMinMaxScaler:
    @pytest.mark.parametrize("minimum, maximum", [(1, 0), (0, 0), (1, 1)])
    def test_invalid_min_max(self, minimum: float, maximum: float) -> None:
        with pytest.raises(ValueError):
            MinMaxScaler(minimum, maximum)

    @pytest.mark.parametrize("x", [0, 1, 0.5, 10, -1, -0.5, -10])
    def test_encode_decode(self, x: float) -> None:
        scaler = MinMaxScaler(0, 1)
        assert scaler.decode(scaler(x)) == x, "Should encode and decode."

    def test_probabilities_predict(self) -> None:
        scaler = MinMaxScaler(0, 1)
        x = torch.rand(1, 10)
        probs = scaler.probabilities_batch(x)
        preds = scaler.predict_batch(probs)
        assert preds == x.squeeze().tolist(), "Should predict the batch."

    def test_majority_vote(self) -> None:
        encoder = MinMaxScaler(0, 1)
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert (
            encoder.majority_vote(x) == 0.3
        ), "Should compute the majority vote."


class TestMultiLabelEncoder:
    labels = ["fizz", "buzz", "jazz"]

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

    def test_probabilities_predict(self) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        x = torch.Tensor([-0.1, 0.9, 0.6])
        probs = encoder.probabilities_batch(x)
        preds = encoder.predict_batch(probs)
        assert preds == [0, 1, 1], "Should predict the batch."

    def test_majority_vote(self) -> None:
        encoder = MultiLabelEncoder(0.5, self.labels)
        x = [["fizz", "jazz"], ["fizz"], ["fizz", "buzz"]]
        assert encoder.majority_vote(x) == [
            "fizz"
        ], "Should compute the majority vote."


class TestLabelEncoder:
    labels = ["fizz", "buzz", "jazz"]

    @pytest.mark.parametrize("label", ["jazz", "buzz", "fizz"])
    def test_encode_decode(self, label: str) -> None:
        encoder = LabelEncoder(self.labels)
        assert (
            encoder.decode(encoder(label)) == label
        ), "Should encode and decode."

    def test_probabilities_predict(self) -> None:
        encoder = LabelEncoder(self.labels)
        x = torch.Tensor([[0, 1, 0], [0, 0, 1]])
        probs = encoder.probabilities_batch(x)
        preds = encoder.predict_batch(probs)
        assert preds == [1, 2], "Should predict the batch."

    def test_majority_vote(self) -> None:
        encoder = LabelEncoder(self.labels)
        x = ["fizz", "buzz", "jazz", "fizz", "buzz", "fizz"]
        assert (
            encoder.majority_vote(x) == "fizz"
        ), "Should compute the majority vote."

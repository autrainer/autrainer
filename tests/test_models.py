from typing import Optional, Tuple, Type, Union

import pytest
import torch

from autrainer.models import (
    FFNN,
    TDNNFFNN,
    W2V2FFNN,
    AbstractModel,
    ASTModel,
    AudioRNNModel,
    Cnn10,
    Cnn14,
    LEAFNet,
    SeqFFNN,
    TimmModel,
    TorchvisionModel,
    WhisperFFNN,
)
from autrainer.models.utils import ExtractLayerEmbeddings


MODEL_FIXTURES = [
    (FFNN, {"input_size": 64, "hidden_size": 128}, (64,)),
    (TDNNFFNN, {"hidden_size": 128}, (16000,)),
    (
        W2V2FFNN,
        {
            "model_name": "facebook/wav2vec2-base",
            "freeze_extractor": True,
            "hidden_size": 128,
        },
        (16000,),
    ),
    (ASTModel, {}, (1024, 128)),
    (AudioRNNModel, {"model_name": "emo18"}, (1, 16000)),
    (Cnn10, {}, (1, 128, 64)),
    (Cnn14, {}, (1, 128, 64)),
    (LEAFNet, {}, (16000,)),
    (
        SeqFFNN,
        {
            "backbone_input_dim": 64,
            "backbone_hidden_size": 128,
            "backbone_num_layers": 2,
            "hidden_size": 128,
        },
        (1, 128, 64),
    ),
    (TimmModel, {"timm_name": "efficientnet_b0"}, (3, 64, 64)),
    (TorchvisionModel, {"torchvision_name": "efficientnet_b0"}, (3, 64, 64)),
    (
        WhisperFFNN,
        {"model_name": "openai/whisper-tiny", "hidden_size": 128},
        (80, 3000),
    ),
]


class TestAllModels:
    @pytest.mark.parametrize(
        ("model_class", "model_kwargs", "input_shape"),
        MODEL_FIXTURES,
    )
    def test_model(
        self,
        model_class: Type[AbstractModel],
        model_kwargs: dict,
        input_shape: Tuple[int],
    ) -> None:
        model = model_class(output_dim=10, transfer=False, **model_kwargs)
        self._test_model(model, input_shape)

    @staticmethod
    def _test_model(
        model: AbstractModel,
        input_shape: Tuple[int],
        eval: bool = True,
        expected: Optional[Tuple[int, ...]] = None,
    ) -> None:
        if eval:
            model.eval()
        x = torch.randn(1, *input_shape)
        if expected is None:
            expected = (1, model.output_dim)
        assert model(x).shape == expected, "Model output shape mismatch"
        emb = model.embeddings(x)
        assert isinstance(emb, torch.Tensor), "Model embeddings not a tensor"

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize(
        ("model_class", "model_kwargs"),
        [
            (AudioRNNModel, {"model_name": "emo18"}),
            (FFNN, {"input_size": 64, "hidden_size": 128}),
            (
                SeqFFNN,
                {
                    "backbone_input_dim": 64,
                    "backbone_hidden_size": 128,
                    "backbone_num_layers": 2,
                    "hidden_size": 128,
                },
            ),
        ],
    )
    def test_invalid_transfer(
        self,
        model_class: Type[AbstractModel],
        model_kwargs: dict,
    ) -> None:
        model_class(output_dim=10, transfer=True, **model_kwargs)


class TestCnn10Cnn14:
    @pytest.mark.parametrize(
        ("cls", "expected"),
        [(Cnn10, (1, 8, 10)), (Cnn14, (1, 4, 10))],
    )
    def test_segmentwise(
        self,
        cls: Type[Union[Cnn10, Cnn14]],
        expected: Tuple[int, int, int],
    ) -> None:
        model = cls(output_dim=10, segmentwise=True)
        TestAllModels._test_model(model, (1, 128, 64), expected=expected)

    def test_invalid_link(self) -> None:
        with pytest.raises(ValueError, match="Failed to download"):
            Cnn10(
                output_dim=10,
                transfer="https://zenodo.org/records/invalid.pt",
            )


class TestAudioRNNModel:
    @pytest.mark.parametrize(
        ("model_name", "cell"),
        [
            ("emo18", "LSTM"),
            ("zhao19", "LSTM"),
            ("emo18", "GRU"),
            ("zhao19", "GRU"),
        ],
    )
    def test_model(self, model_name: str, cell: str) -> None:
        model = AudioRNNModel(output_dim=10, model_name=model_name, cell=cell)
        TestAllModels._test_model(model, (1, 16000))


class TestLEAFNet:
    @pytest.mark.parametrize("mode", ["interspeech", "speech_brain"])
    def test_mode(self, mode: str) -> None:
        model = LEAFNet(output_dim=10, mode=mode)
        TestAllModels._test_model(model, (16000,))

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="'interspeech' and 'speech_brain'"):
            LEAFNet(output_dim=10, mode="invalid")

    def test_invalid_efficientnet(self) -> None:
        with pytest.raises(ValueError, match="Only EfficientNet models"):
            LEAFNet(output_dim=10, efficientnet_type="resnet18")

    @pytest.mark.parametrize(
        "initialization",
        ["mel", "bark", "linear-constant", "constant", "uniform", "zeros"],
    )
    def test_initializations(self, initialization: str) -> None:
        model = LEAFNet(output_dim=10, initialization=initialization)
        TestAllModels._test_model(model, (16000,))

    def test_invalid_initialization(self) -> None:
        with pytest.raises(ValueError, match="got initialization"):
            LEAFNet(output_dim=10, initialization="invalid")


class TestSeqFFNN:
    def _mock_default_kwargs(self) -> dict:
        return {
            "output_dim": 10,
            "backbone_input_dim": 64,
            "backbone_hidden_size": 128,
            "backbone_num_layers": 2,
            "hidden_size": 128,
        }

    @pytest.mark.parametrize(
        ("cell", "bidirectional"),
        [
            ("LSTM", False),
            ("GRU", False),
            ("LSTM", True),
            ("GRU", True),
        ],
    )
    def test_backbone_cell(self, cell: str, bidirectional: bool) -> None:
        model = SeqFFNN(
            **self._mock_default_kwargs(),
            backbone_cell=cell,
            backbone_bidirectional=bidirectional,
        )
        TestAllModels._test_model(model, (1, 128, 64))

    def test_time_pooling(self) -> None:
        model = SeqFFNN(
            **self._mock_default_kwargs(),
            backbone_time_pooling=False,
        )
        TestAllModels._test_model(model, (1, 128, 64), expected=(1, 128, 10))

    def test_invalid_cell(self) -> None:
        with pytest.raises(NotImplementedError):
            SeqFFNN(**self._mock_default_kwargs(), backbone_cell="RNN")


class TestTorchvisionModel:
    def test_transfer(self) -> None:
        model = TorchvisionModel(
            output_dim=10,
            torchvision_name="efficientnet_b0",
            transfer=True,
        )
        TestAllModels._test_model(model, (3, 64, 64))


class TestExtractLayerEmbeddings:
    def test_missing_linear_layer(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
        )
        with pytest.raises(ValueError, match="model does not contain a linear layer"):
            ExtractLayerEmbeddings(model)

    def test_invalid_module_length(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 10),
        )
        with pytest.raises(ValueError, match="must have at least two layers"):
            ExtractLayerEmbeddings(model)

    def test_embedding_hook(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )
        embeddings = ExtractLayerEmbeddings(model)
        embeddings._register()
        embeddings._register()
        embeddings._unregister()
        embeddings._unregister()
        x = torch.randn(1, 64)
        assert embeddings(x).shape == (1, 128), "Embeddings shape mismatch"


class TestWrongArgumentModel:
    @pytest.mark.xfail(raises=NameError)
    def test_wrong_arguments(self) -> None:
        class Foo(AbstractModel):
            def __init__(self, output_dim: int) -> None:
                super().__init__(output_dim, None)

            def embeddings(self, x: torch.Tensor) -> torch.Tensor:
                return super().embeddings(x)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        bar = Foo(1)
        bar.inputs  # noqa: B018

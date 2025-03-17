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
from autrainer.models.crnn import CRNN
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
    (CRNN, {}, (1, 500, 64)),
]


class TestAllModels:
    @pytest.mark.parametrize(
        "model_class, model_kwargs, input_shape",
        MODEL_FIXTURES,
    )
    def test_model(
        self,
        model_class: Type[AbstractModel],
        model_kwargs: dict,
        input_shape: Tuple[int],
    ) -> None:
        model = model_class(output_dim=10, **model_kwargs)
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


class TestASTModel:
    def test_transfer(self) -> None:
        model = ASTModel(
            output_dim=10,
            transfer="MIT/ast-finetuned-audioset-10-10-0.4593",
        )
        TestAllModels._test_model(model, (1024, 128))


class TestCnn10Cnn14:
    @pytest.mark.parametrize(
        "cls, link",
        [
            (
                Cnn10,
                "https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth",
            ),
            (
                Cnn14,
                "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth",
            ),
        ],
    )
    def test_transfer(self, cls: Type[Union[Cnn10, Cnn14]], link: str) -> None:
        model = cls(output_dim=10, transfer=link)
        TestAllModels._test_model(model, (1, 128, 64))

    @pytest.mark.parametrize(
        "cls, expected",
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
        with pytest.raises(ValueError):
            Cnn10(
                output_dim=10,
                transfer="https://zenodo.org/records/invalid.pt",
            )


class TestAudioRNNModel:
    @pytest.mark.parametrize(
        "model_name, cell",
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
        with pytest.raises(ValueError):
            LEAFNet(output_dim=10, mode="invalid")

    def test_invalid_efficientnet(self) -> None:
        with pytest.raises(ValueError):
            LEAFNet(output_dim=10, efficientnet_type="resnet18")

    @pytest.mark.parametrize(
        "initialization",
        ["mel", "bark", "linear-constant", "constant", "uniform", "zeros"],
    )
    def test_initializations(self, initialization: str) -> None:
        model = LEAFNet(output_dim=10, initialization=initialization)
        TestAllModels._test_model(model, (16000,))

    def test_invalid_initialization(self) -> None:
        with pytest.raises(ValueError):
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
        "cell, bidirectional",
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


class TestCRNN:
    @pytest.mark.parametrize(
        "test_case, params",
        [
            ("activation_relu", {"activation": "relu"}),
            ("activation_leakyrelu", {"activation": "leakyrelu"}),
            ("activation_glu", {"activation": "glu"}),
            ("activation_cg", {"activation": "cg"}),
            ("cnn_single_layer", {"n_cnn_layers": 1, "kernel_size": 3}),
            ("cnn_double_layer", {"n_cnn_layers": 2, "kernel_size": 5}),
            (
                "cnn_multi_layer",
                {"n_cnn_layers": 4, "kernel_size": 3, "pooling": [1, 2]},
            ),
            (
                "cnn_varying_kernels",
                {
                    "n_cnn_layers": 3,
                    "kernel_size": [3, 3, 5],
                    "pooling": [1, 2],
                },
            ),
            ("rnn_small", {"hidden_size": 32, "n_layers_rnn": 1}),
            ("rnn_medium", {"hidden_size": 64, "n_layers_rnn": 2}),
            ("rnn_large", {"hidden_size": 128, "n_layers_rnn": 3}),
            ("filters_small", {"nb_filters": 32, "pooling": [2, 2]}),
            ("filters_medium", {"nb_filters": 64, "pooling": [4, 4]}),
            (
                "filters_varying",
                {
                    "nb_filters": [32, 64, 128],
                    "pooling": [[2, 2], [2, 2], [2, 2]],
                },
            ),
            ("dropout_none", {"dropout": 0.0}),
            ("dropout_moderate", {"dropout": 0.3}),
            ("dropout_varying", {"dropout": [0.1, 0.2, 0.3]}),
            ("stride_padding_small", {"stride": 1, "padding": 1}),
            ("stride_padding_medium", {"stride": 1, "padding": 2}),
            (
                "stride_padding_varying",
                {"stride": [1, 1, 1], "padding": [1, 2, 1]},
            ),
            ("channels_1", {"in_channels": 1}),
            ("channels_2", {"in_channels": 2}),
            ("channels_3", {"in_channels": 3}),
            ("attention_true", {"attention": True}),
            ("attention_false", {"attention": False}),
            (
                "complex_combo_1",
                {
                    "activation": "leakyrelu",
                    "n_cnn_layers": 2,
                    "kernel_size": 5,
                    "hidden_size": 64,
                    "n_layers_rnn": 2,
                    "dropout": [0.1, 0.2, 0.3],
                    "attention": True,
                },
            ),
            (
                "complex_combo_2",
                {
                    "activation": "glu",
                    "in_channels": 2,
                    "n_cnn_layers": 3,
                    "kernel_size": [3, 3, 5],
                    "nb_filters": [32, 64, 128],
                    "pooling": [[1, 2], [1, 2], [1, 2]],
                    "hidden_size": 32,
                    "attention": False,
                },
            ),
            (
                "complex_combo_3",
                {
                    "activation": "cg",
                    "n_cnn_layers": 2,
                    "stride": [1, 1],
                    "padding": [2, 1],
                    "nb_filters": 32,
                    "n_layers_rnn": 3,
                    "dropout": 0.2,
                },
            ),
        ],
    )
    def test_model_configurations(self, test_case: str, params: dict) -> None:
        input_shape = (1, 500, 64)
        if "in_channels" in params:
            input_shape = (params["in_channels"], 500, 64)
        model = CRNN(output_dim=10, **params)
        TestAllModels._test_model(model, input_shape)


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
        with pytest.raises(ValueError):
            ExtractLayerEmbeddings(model)

    def test_invalid_module_length(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 10),
        )
        with pytest.raises(ValueError):
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

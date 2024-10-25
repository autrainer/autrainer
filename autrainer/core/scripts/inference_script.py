from dataclasses import dataclass
import glob
import os
from typing import Optional

import autrainer

from .abstract_script import AbstractScript, MockParser
from .command_line_error import CommandLineError
from .utils import catch_cli_errors


@dataclass
class InferenceArgs:
    model: str
    input: str
    output: str
    checkpoint: str
    device: str
    extension: str
    recursive: bool
    embeddings: bool
    update_frequency: int
    preprocess_cfg: Optional[str]
    window_length: Optional[float]
    stride_length: Optional[float]
    min_length: Optional[float]
    sample_rate: Optional[int]


class InferenceScript(AbstractScript):
    def __init__(self) -> None:
        e = (
            "Example: autrainer inference hf:autrainer/example "
            "input/ output/ -d cuda:0"
        )
        super().__init__(
            "inference",
            "Perform inference on a trained model.",
            epilog=e,
            dataclass=InferenceArgs,
        )

    def add_arguments(self) -> None:
        self.parser.add_argument(
            "model",
            type=str,
            help=(
                "Local path to model directory or Hugging Face link of the "
                "format: `hf:repo_id[@revision][:subdir]#local_dir`. "
                "Should contain at least one state subdirectory, the "
                "`model.yaml`, `file_handler.yaml`, `target_transform.yaml`, "
                "and `inference_transform.yaml` files."
            ),
        )
        self.parser.add_argument(
            "input",
            type=str,
            help=(
                "Path to input directory. Should contain audio files "
                "of the specified extension."
            ),
        )
        self.parser.add_argument(
            "output",
            type=str,
            help=(
                "Path to output directory. Output includes a YAML file with "
                "predictions and a CSV file with model outputs."
            ),
        )
        self.parser.add_argument(
            "-c",
            "--checkpoint",
            type=str,
            metavar="C",
            required=False,
            default="_best",
            help=(
                "Checkpoint to use for evaluation. "
                "Defaults to '_best' (on validation set)."
            ),
        )
        self.parser.add_argument(
            "-d",
            "--device",
            type=str,
            metavar="D",
            required=False,
            default="cpu",
            help=(
                "CUDA-enabled device to use for processing. "
                "Defaults to 'cpu'."
            ),
        )
        self.parser.add_argument(
            "-e",
            "--extension",
            type=str,
            metavar="E",
            required=False,
            default="wav",
            help=(
                "Type of file to look for in the input directory. "
                "Defaults to 'wav'."
            ),
        )
        self.parser.add_argument(
            "-r",
            "--recursive",
            action="store_true",
            help=(
                "Recursively search for files in the input directory. "
                "Defaults to False."
            ),
        )
        self.parser.add_argument(
            "-emb",
            "--embeddings",
            action="store_true",
            help=(
                "Extract embeddings from the model in addition to predictions."
                "For each file, a .pt file with embeddings will be saved."
                "Defaults to False."
            ),
        )
        self.parser.add_argument(
            "-u",
            "--update-frequency",
            type=int,
            default=1,
            metavar="F",
            required=False,
            help=(
                "Frequency of progress bar updates. "
                "If 0, the progress bar will be disabled. Defaults to 1."
            ),
        )
        self.parser.add_argument(
            "-p",
            "--preprocess-cfg",
            type=str,
            metavar="P",
            required=False,
            default="default",
            help=(
                "Preprocessing configuration to apply to input. Can be a path "
                "to a YAML file or the name of the preprocessing configuration "
                "in the local or autrainer 'conf/preprocessing' directory. "
                "If 'default', the default preprocessing configuration used "
                "during training will be applied. If 'None', no preprocessing "
                "will be applied. Defaults to 'default'."
            ),
        )
        self.parser.add_argument(
            "-w",
            "--window-length",
            type=float,
            metavar="W",
            required=False,
            default=None,
            help=(
                "Window length for sliding window inference in seconds. "
                "If None, the entire input will be processed at once. "
                "Defaults to None."
            ),
        )
        self.parser.add_argument(
            "-s",
            "--stride-length",
            type=float,
            metavar="S",
            required=False,
            default=None,
            help=(
                "Stride length for sliding window inference in seconds. "
                "If None, the entire input will be processed at once. "
                "Defaults to None."
            ),
        )
        self.parser.add_argument(
            "-m",
            "--min-length",
            type=float,
            metavar="M",
            required=False,
            default=None,
            help=(
                "Minimum length of audio file to process in seconds. "
                "Files shorter than the minimum length are padded with zeros. "
                "Sample rate has to be specified for padding. "
                "If None, no minimum length is enforced. Defaults to None."
            ),
        )
        self.parser.add_argument(
            "-sr",
            "--sample-rate",
            type=int,
            metavar="SR",
            required=False,
            default=None,
            help=(
                "Sample rate of audio files in Hz. Has to be specified for "
                "sliding window inference. Defaults to None."
            ),
        )

    def main(self, args: InferenceArgs) -> None:
        args.model = self._assert_model_exists(args)
        self._assert_valid_input(args)
        self._assert_valid_device(args)
        self._assert_valid_checkpoint(args)
        self._assert_and_set_preprocess_path(args)
        self._inference(args)

    def _assert_model_exists(self, args: InferenceArgs) -> str:
        from autrainer.serving import get_model_path

        try:
            return get_model_path(args.model)
        except ValueError as e:
            raise CommandLineError(self.parser, str(e), code=1)

    def _assert_checkpoint_exists(self, args: InferenceArgs) -> None:
        path = os.path.join(args.model, args.checkpoint, "model.pt")
        if os.path.exists(path):
            return
        raise CommandLineError(
            self.parser,
            f"Checkpoint '{args.checkpoint}' does not exist.",
            code=1,
        )

    def _assert_valid_input(self, args: InferenceArgs) -> None:
        if not os.path.exists(args.input):
            raise CommandLineError(
                self.parser,
                f"Input '{args.input}' does not exist.",
                code=1,
            )
        if not os.path.isdir(args.input):
            raise CommandLineError(
                self.parser,
                f"Input '{args.input}' is not a directory.",
                code=1,
            )

        pattern = (
            f"**/*.{args.extension}"
            if args.recursive
            else f"*.{args.extension}"
        )
        if not glob.glob(
            os.path.join(args.input, pattern),
            recursive=True,
        ):
            raise CommandLineError(
                self.parser,
                f"No '{args.extension}' files found in '{args.input}'.",
                code=1,
            )

    def _assert_valid_device(self, args: InferenceArgs) -> None:
        if args.device == "cpu" or args.device.startswith("cuda:"):
            return
        raise CommandLineError(
            self.parser,
            (
                f"Invalid device '{args.device}'. "
                "Please specify 'cpu' or 'cuda:<device_id>'."
            ),
        )

    def _assert_valid_checkpoint(self, args: InferenceArgs) -> None:
        path = os.path.join(args.model, args.checkpoint, "model.pt")
        if os.path.exists(path):
            return
        raise CommandLineError(
            self.parser,
            f"Checkpoint '{args.checkpoint}' does not exist.",
            code=1,
        )

    def _assert_and_set_preprocess_path(self, args: InferenceArgs) -> None:
        if args.preprocess_cfg == "default":
            return

        if args.preprocess_cfg == "None" or args.preprocess_cfg is None:
            args.preprocess_cfg = None
            return

        if not args.preprocess_cfg.endswith(".yaml"):
            args.preprocess_cfg += ".yaml"

        if os.path.exists(args.preprocess_cfg):
            return

        local_path = os.path.join("conf", "preprocessing", args.preprocess_cfg)
        if os.path.exists(local_path):
            args.preprocess_cfg = local_path
            return

        lib_path = os.path.join(
            os.path.dirname(autrainer.__path__[0]),
            "autrainer-configurations",
            "preprocessing",
            args.preprocess_cfg,
        )
        if os.path.exists(lib_path):
            args.preprocess_cfg = lib_path
            return

        raise CommandLineError(
            self.parser,
            (
                f"Preprocessing configuration '{args.preprocess_cfg}' "
                "does not exist."
            ),
            code=1,
        )

    def _inference(self, args: InferenceArgs) -> None:
        from autrainer.serving import Inference

        inference = Inference(
            model_path=args.model,
            checkpoint=args.checkpoint,
            device=args.device,
            preprocess_cfg=args.preprocess_cfg,
            window_length=args.window_length,
            stride_length=args.stride_length,
            min_length=args.min_length,
            sample_rate=args.sample_rate,
        )

        results = inference.predict_directory(
            args.input,
            extension=args.extension,
            recursive=args.recursive,
            update_frequency=args.update_frequency,
        )
        inference.save_prediction_yaml(results, args.output)
        inference.save_prediction_results(results, args.output)

        if not args.embeddings:
            return

        results = inference.embed_directory(
            args.input,
            extension=args.extension,
            recursive=args.recursive,
            update_frequency=args.update_frequency,
        )
        inference.save_embeddings(
            results,
            args.output,
            input_extension=args.extension,
        )


@catch_cli_errors
def inference(
    model: str,
    input: str,
    output: str,
    checkpoint: str = "_best",
    device: str = "cpu",
    extension: str = "wav",
    recursive: bool = False,
    embeddings: bool = False,
    update_frequency: int = 1,
    preprocess_cfg: Optional[str] = "default",
    window_length: Optional[float] = None,
    stride_length: Optional[float] = None,
    min_length: Optional[float] = None,
    sample_rate: Optional[int] = None,
) -> None:
    """Perform inference on a trained model.

    If called in a notebook, the function will not raise an error and print
    the error message instead.

    Args:
        model: Local path to model directory or Hugging Face link of the
            format: `hf:repo_id[@revision][:subdir]#local_dir`.
            Should contain at least one state subdirectory, the
            `model.yaml`, `file_handler.yaml`, `target_transform.yaml`,
            and `inference_transform.yaml` files.
        input: Path to input directory. Should contain audio files of the
            specified extension.
        output: Path to output directory. Output includes a YAML file with
            predictions and a CSV file with model outputs.
        checkpoint: Checkpoint to use for evaluation. Defaults to '_best'.
        device: CUDA-enabled device to use for processing. Defaults to 'cpu'.
        extension: Type of file to look for in the input directory.
            Defaults to 'wav'.
        recursive: Recursively search for files in the input directory.
            Defaults to False.
        embeddings: Extract embeddings from the model in addition to
            predictions. For each file, a .pt file with embeddings will be
            saved. Defaults to False.
        update_frequency: Frequency of progress bar updates. If 0, the progress
            bar will be disabled. Defaults to 1.
        preprocess_cfg: Preprocessing configuration to apply to input. Can be
            a path to a YAML file or the name of the preprocessing
            configuration in the local or autrainer 'conf/preprocessing'
            directory. If "default", the default preprocessing configuration
            used during training will be applied. If None, no preprocessing
            will be applied. Defaults to "default".
        window_length: Window length for sliding window inference in seconds.
            If None, the entire input will be processed at once.
            Defaults to None.
        stride_length: Stride length for sliding window inference in seconds.
            If None, the entire input will be processed at once.
            Defaults to None.
        min_length: Minimum length of audio file to process in seconds.
            Files shorter than the minimum length are padded with zeros.
            Sample rate has to be specified for padding. If None, no minimum
            length is enforced. Defaults to None.
        sample_rate: Sample rate of audio files in Hz. Has to be specified for
            sliding window inference. Defaults to None.

    Raises:
        CommandLineError: If the model, input, or preprocessing configuration
            does not exist, or if the device is invalid.
    """
    script = InferenceScript()
    script.parser = MockParser()
    script.main(
        InferenceArgs(
            model,
            input,
            output,
            checkpoint,
            device,
            extension,
            recursive,
            embeddings,
            update_frequency,
            preprocess_cfg,
            window_length,
            stride_length,
            min_length,
            sample_rate,
        )
    )

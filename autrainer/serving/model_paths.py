from abc import ABC, abstractmethod
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple


MODEL_FILES = [
    "model.yaml",
    "file_handler.yaml",
    "target_transform.yaml",
    "inference_transform.yaml",
    "preprocess_file_handler.yaml",
    "preprocess_pipeline.yaml",
]
SAVE_FILES = tuple(["model.pt"] + MODEL_FILES)


class AbstractModelPath(ABC):
    def __init__(self, model_path: str) -> None:
        """Abstract class for model path construction.

        Args:
            model_path: Model path.
        """
        self.model_path = model_path

    @abstractmethod
    def verify(self) -> None:
        """Verify the model path.

        Raises:
            ValueError: If the model path is invalid.
        """

    @abstractmethod
    def create_model_path(self) -> str:
        """Create the directory of the model.

        Returns:
            The directory of the model.
        """

    def get_model_path(self) -> str:
        """Get the directory of the model.

        Returns:
            The directory of the model.
        """
        path = self.create_model_path()
        for file in MODEL_FILES:
            if not os.path.isfile(os.path.join(path, file)):
                raise ValueError(
                    f"Invalid model directory: '{file}' not found in '{path}'"
                )
        return path


class LocalModelPath(AbstractModelPath):
    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        self.verify()

    def verify(self) -> None:
        if not os.path.isdir(self.model_path):
            raise ValueError(
                f"Invalid local model directory: '{self.model_path}'"
            )

    def create_model_path(self) -> str:
        return self.model_path


class HubModelPath(AbstractModelPath):
    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        self.verify()

    def verify(self) -> None:
        """Verify the given hugging face model path.

        Verification includes checking the validity of the repository, revision,
        and repository subdirectory in the model path.
        The model files are also verified to ensure that a single model
        directory is found.

        Raises:
            ValueError: If the model path is invalid.
            ValueError: If the repository, revision or repository subdirectory
            is not found.
            ValueError: If the model directory is not found or multiple model
            directories are found.
        """
        import torch

        repo_id, revision, subdir, local_dir = self._parse_hub_path()
        files = self._verify_hub_path(repo_id, revision, subdir)

        model_files = [f for f in files if os.path.basename(f) == "model.yaml"]

        if subdir:
            model_files = [f for f in model_files if f.startswith(subdir)]

        self._verify_model_files(repo_id, revision, subdir, model_files)

        model_file_loc = os.path.dirname(model_files[0])

        self.repo_id = repo_id
        self.revision = revision
        self.local_dir = Path(
            local_dir
            or os.path.join(
                torch.hub.get_dir(),
                "autrainer",
                repo_id.replace("/", "--") + f"--{revision or 'main'}",
            )
        ).as_posix()
        self.files = [
            Path(f).relative_to(model_file_loc).as_posix()
            for f in files
            if f.startswith(model_file_loc) and f.endswith(SAVE_FILES)
        ]
        self.subdir = model_file_loc

    def create_model_path(self) -> str:
        """Create the local directory of the model.

        The model is downloaded to the local directory if it is not already
        present.
        If no local directory is specified, the model is downloaded to the
        default torch.hub directory.

        Returns:
            Local directory of the model.
        """
        self._download()
        return os.path.join(self.local_dir, self.subdir)

    def _parse_hub_path(
        self,
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        # pattern: hf:repo_id[@revision][:subdir]#local_dir
        pattern = r"^hf:([^@:#]+)(?:@([^:#]+))?(?::([^#]+))?(?:#(.+))?$"

        match = re.match(pattern, self.model_path)
        if not match:
            raise ValueError(
                f"Invalid hugging face path format: '{self.model_path}'"
            )

        return match.groups()

    @staticmethod
    def _verify_hub_path(
        repo_id: str,
        revision: Optional[str],
        subdir: Optional[str],
    ) -> List[str]:
        from huggingface_hub import HfApi

        try:
            hf_api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
            files = hf_api.list_repo_files(
                repo_id,
                revision=revision,
            )
        except Exception:
            raise ValueError(
                f"Invalid hugging face repo id: '{repo_id}'"
                + (f" or revision: '{revision}'" if revision else "")
            )
        if subdir is not None and f"{subdir}/model.yaml" not in files:
            raise ValueError(
                f"Subdirectory '{subdir}' not found in repo '{repo_id}' "
                "or is not a valid model directory"
            )
        return files

    @staticmethod
    def _verify_model_files(
        repo_id: str,
        revision: Optional[str],
        subdir: Optional[str],
        model_files: List[str],
    ) -> None:
        if not model_files:
            raise ValueError(
                f"No model directory found in repo '{repo_id}'"
                + (f" at revision '{revision}'" if revision else "")
                + (f" with subdir '{subdir}'" if subdir else "")
            )
        if len(model_files) > 1:
            raise ValueError(
                f"Multiple model directories found in repo '{repo_id}'"
            )

    def _download(self) -> None:
        from huggingface_hub import HfApi
        from tqdm import tqdm

        from autrainer.core.utils import silence

        os.makedirs(self.local_dir, exist_ok=True)
        self.files = [
            f
            for f in self.files
            if not os.path.isfile(os.path.join(self.local_dir, self.subdir, f))
        ]
        if not self.files:
            return
        hf_api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
        pbar = tqdm(self.files, desc="Downloading:")
        for file in pbar:
            pbar.set_description(f"Downloading: {file}")
            with silence():
                hf_api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=file,
                    subfolder=self.subdir,
                    revision=self.revision,
                    local_dir=self.local_dir,
                )


def get_model_path(model_path: str) -> str:
    if model_path.startswith("hf:"):
        return HubModelPath(model_path).get_model_path()
    return LocalModelPath(model_path).get_model_path()

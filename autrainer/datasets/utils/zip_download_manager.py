from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile

import requests
from tqdm import tqdm


class ZipDownloadManager:
    def __init__(
        self,
        files: Dict[str, str],
        path: str,
        max_threads: int = 4,
    ) -> None:
        """Download and extract zip files.

        Args:
            files: Dictionary of filenames and URLs.
            path: Path to download and extract the files to.
            max_threads: Maximum number of threads to use. Defaults to 4.
        """
        self.files = files
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_threads = max_threads

    def download(self, check_exist: Optional[List[str]] = None) -> None:
        """Download all files in `files` to `path`.

        Skips files if they are already present or all files in `check_exist`
        are present.

        Args:
            check_exist: List of filenames to check if they already exist.
                Defaults to None.
        """
        if self._check_exist(check_exist):
            print("All files already exist, skipping download.")
            return
        print(f"Downloading {len(self.files)} files to '{self.path}' ...")

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(self._download_file, filename, url)
                for filename, url in self.files.items()
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error downloading or extracting file: {e}")

        print()

    def extract(self, check_exist: Optional[List[str]] = None) -> None:
        """Extract all files in `files` to `path`.

        Does not extract files if all files in `check_exist` are present.

        Args:
            check_exist: List of filenames to check if they already exist.
                Defaults to None.
        """
        if self._check_exist(check_exist):
            print("All files already exist, skipping extraction.")
            return
        print(f"Extracting {len(self.files)} files to '{self.path}' ...")
        for filename in self.files.keys():
            self._extract_zip(self.path / filename, self.path)

    def _check_exist(self, check_exist: Optional[List[str]] = None) -> bool:
        return check_exist and all(
            (self.path / filename).exists() for filename in check_exist
        )

    def _download_file(self, filename: str, url: str) -> None:
        dest = self.path / filename
        if dest.exists():
            print(f"{dest.name} already exists, skipping download.")
            return
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(dest, "wb") as file,
            tqdm(
                desc=str(dest.name),
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def _extract_zip(self, zip_path: Path, extract_to: Path) -> None:
        if not zip_path.suffix == ".zip":
            return
        with ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)

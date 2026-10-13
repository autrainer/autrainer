import os
import subprocess
import sys


def save_requirements(output_directory: str) -> None:
    reqs_output = os.path.join(output_directory, "requirements.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )

    with open(reqs_output, "w", encoding="utf-8") as f:
        f.write(result.stdout)

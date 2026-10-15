from importlib.metadata import distributions
from pathlib import Path


def save_requirements(output_directory: str) -> None:
    reqs = sorted(
        {
            f"{name}=={d.version}"
            for d in distributions()
            if (name := d.metadata["Name"])
        },
        key=str.lower,
    )
    p = Path(output_directory) / "requirements.txt"
    p.write_text("\n".join(reqs) + "\n", encoding="utf-8")

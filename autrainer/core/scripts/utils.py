from functools import wraps
import sys
from typing import Any, Callable, Dict, Optional, TypeVar

from .command_line_error import CommandLineError


F = TypeVar("F", bound=Callable[..., Any])
TQDM_LINE_ENDINGS = ["it/s]", "s/it]", "B/s]"]


def encode_override_kwargs(override_kwargs: Dict[str, Any]) -> Dict[str, str]:
    """Encode Hydra override kwargs by converting all values to strings.
    List values are converted to strings by joining the elements with commas.
    Boolean values are converted to lowercase strings.
    Duplicate keys are automatically overwritten.

    Args:
        override_kwargs: The override kwargs to encode.

    Returns:
        Encoded override kwargs.

    Raises:
        CommandLineError: If any override argument values are not of type list,
            str, int, float, or bool.
    """
    encoded_kwargs = {}
    for key, value in override_kwargs.items():
        if isinstance(value, list):
            value = ",".join(map(str, value))
        elif isinstance(value, bool):
            value = str(value).lower()
        if not isinstance(value, (str, int, float)):
            raise CommandLineError(
                f"Hydra override argument values must be of type list, "
                f"str, int, float, or bool, but got {type(value)}."
            )
        encoded_kwargs[key] = str(value)
    return encoded_kwargs


def run_hydra_cmd(
    cmd: str,
    override_kwargs: Optional[dict] = None,
    config_name: str = "config",
    config_path: Optional[str] = None,
    cmd_prefix: str = "autrainer",
) -> None:
    """Run a Hydra command in a subprocess and print the output.

    Args:
        cmd: The Hydra command to run.
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "config".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
        cmd_prefix: The command prefix to use. Defaults to "autrainer".
    """
    import subprocess

    cmd = f"{cmd_prefix} {cmd} -cn {config_name}.yaml"
    if config_path is not None:
        cmd += f" -cp {config_path}"
    if override_kwargs is not None:
        for key, value in encode_override_kwargs(override_kwargs).items():
            cmd += f" {key}={value}"

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        universal_newlines=False,
    ) as process:
        tqdm_line = False
        for line in process.stdout:
            line = line.rstrip()

            # TODO: find a better way to handle tqdm output
            if any(line.endswith(ending) for ending in TQDM_LINE_ENDINGS):
                print("\r" + line, end="")
                tqdm_line = True
            elif tqdm_line:
                print("\n" + line)
                tqdm_line = False
            else:
                print(line)

    process.stdout.close()
    process.wait()


def running_in_notebook() -> bool:
    return any("jupyter" in arg or "ipykernel" in arg for arg in sys.argv)


def catch_cli_errors(func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except CommandLineError as e:
            if not running_in_notebook():
                raise e
            print(e.message, file=sys.stderr)

    return wrapper


def add_hydra_args_to_sys(
    override_kwargs: Optional[dict] = None,
    config_name: str = "config",
    config_path: Optional[str] = None,
) -> None:
    if config_name.replace(".yaml", "") != "config":
        sys.argv.extend(["-cn", config_name])
    if config_path is not None:
        sys.argv.extend(["-cp", config_path])
    if override_kwargs is not None:
        sys.argv.extend(
            f"{k}={v}"
            for k, v in encode_override_kwargs(override_kwargs).items()
        )

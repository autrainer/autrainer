from typing import Optional

from .command_line_error import CommandLineError


TQDM_LINE_ENDINGS = ["it/s]", "s/it]", "B/s]"]


def run_autrainer_hydra_cmd(
    cmd: str,
    override_kwargs: Optional[dict] = None,
    config_name: str = "config",
    config_path: Optional[str] = None,
) -> None:
    """Run an autrainer Hydra command in a subprocess and print the output.

    Args:
        cmd: The autrainer Hydra command to run.
        override_kwargs: Additional Hydra override arguments to pass to the
            train script.
        config_name: The name of the config (usually the file name without the
            .yaml extension). Defaults to "config".
        config_path: The config path, a directory where Hydra will search for
            config files. If config_path is None no directory is added to the
            search path. Defaults to None.
    """
    import subprocess

    cmd = f"autrainer {cmd} -cn {config_name}.yaml"
    if config_path is not None:
        cmd += f" -cp {config_path}"
    if override_kwargs is not None:
        for key, value in override_kwargs.items():
            if isinstance(value, list):
                value = ",".join(map(str, value))
            if not isinstance(value, (str, int, float, bool)):
                raise CommandLineError(
                    f"Hydra override argument values must be of type list, "
                    f"str, int, float, or bool, but got {type(value)}."
                )
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

import os
from typing import List

from omegaconf import DictConfig
import yaml


def get_run_names(training_directory: str):
    run_names = []
    for dir in os.listdir(training_directory):
        if os.path.isdir(os.path.join(training_directory, dir)):
            run_names.append(dir)
    return run_names


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data: dict):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def get_training_type(training_directory: str, runs: List[str]):
    training_types = []
    for run in runs:
        path = os.path.join(training_directory, run, ".hydra", "config.yaml")
        config = DictConfig(load_yaml(path))
        training_types.append(config.training_type)
    assert (
        len(set(training_types)) == 1
    ), f"Multiple training types found: {set(training_types)}"
    return training_types[0]


def get_plotting_params(training_directory: str, run: str):
    path = os.path.join(training_directory, run, ".hydra", "config.yaml")
    return DictConfig(load_yaml(path)).plotting

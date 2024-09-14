NAMING_CONVENTION = [
    "dataset",
    "model",
    "optimizer",
    "learning_rate",
    "batch_size",
    "training_type",
    "iterations",
    "scheduler",
    "augmentation",
    "seed",
]
"""Naming convention of runs."""

INVALID_AGGREGATIONS = [
    "training_type",
]
"""Invalid aggregations for postprocessing."""

VALID_AGGREGATIONS = list(set(NAMING_CONVENTION) - set(INVALID_AGGREGATIONS))
"""Valid aggregations for postprocessing."""

CONFIG_FOLDERS = [
    "augmentation",
    "dataset",
    "model",
    "optimizer",
    "plotting",
    "preprocessing",
    "scheduler",
    "sharpness",
]
"""Configuration folders for Hydra configurations."""

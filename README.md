<div align="center">
  <img src="https://autrainer.github.io/autrainer/_images/logo_banner.png" alt="autrainer — A Modular and Extensible Deep Learning Toolkit for Computer Audition Tasks">
</div>

# autrainer

[![autrainer PyPI Version](https://img.shields.io/pypi/v/autrainer?logo=pypi&logoColor=b4befe&color=b4befe)](https://pypi.org/project/autrainer/)
[![autrainer Python Versions](https://img.shields.io/pypi/pyversions/autrainer?logo=python&logoColor=b4befe&color=b4befe)](https://pypi.org/project/autrainer/)
[![autrainer Hugging Face](https://img.shields.io/badge/Hugging_Face-autrainer-b4befe?logo=huggingface&logoColor=b4befe)](https://huggingface.co/autrainer)
[![autrainer GitHub License](https://img.shields.io/badge/license-MIT-b4befe?logo=c)](https://github.com/autrainer/autrainer/blob/main/LICENSE)

A Modular and Extensible Deep Learning Toolkit for Computer Audition Tasks.

_autrainer_ is built on top of [PyTorch](https://pytorch.org/) and [Hydra](https://hydra.cc/),
offering a modular and extensible way to perform reproducible deep learning experiments
for computer audition tasks using YAML configuration files and the command line.

## Installation

To install _autrainer_, first ensure that PyTorch (along with torchvision and torchaudio) version 2.0 or higher is installed.
For installation instructions, refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

It is recommended to install _autrainer_ within a virtual environment.
To create a new virtual environment, refer to the [Python venv documentation](https://docs.python.org/3/library/venv.html).

Next, install _autrainer_ using _pip_.

```bash
pip install autrainer
```

The following optional dependencies can be installed to enable additional features:

- `latex` for LaTeX plotting (requires a LaTeX installation).
- `mlflow` for [MLflow](https://mlflow.org/) logging.
- `tensorboard` for [TensorBoard](https://www.tensorflow.org/tensorboard) logging.
- `opensmile` for audio feature extraction with [openSMILE](https://audeering.com/opensmile/).
- `albumentations` for image augmentations with [Albumentations](https://albumentations.ai/).
- `audiomentations` for audio augmentations with [audiomentations](https://github.com/iver56/audiomentations).
- `torch-audiomentations` for audio augmentations with [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations).

To install _autrainer_ with all optional dependencies, use the following command:

```bash
pip install autrainer[all]
```

To install _autrainer_ from source, refer to the [contribution guide](https://autrainer.github.io/autrainer/development/contributing.html).

## Next Steps

To get started using _autrainer_, the [quickstart guide](https://autrainer.github.io/autrainer/usage/quickstart.html) outlines the creation of a simple training configuration
and [tutorials](https://autrainer.github.io/autrainer/usage/tutorials.html) provide examples for implementing custom modules including their configurations.

For a complete list of available CLI commands, refer to the [CLI reference](https://autrainer.github.io/autrainer/usage/cli_reference.html) or the [CLI wrapper](https://autrainer.github.io/autrainer/usage/cli_wrapper.html).

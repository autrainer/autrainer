.. image:: https://autrainer.github.io/autrainer/_static/logo_banner.png
    :alt: autrainer — A Modular and Extensible Deep Learning Toolkit for Computer Audition Tasks
    :align: center


autrainer
=========

|pypi| |python_versions| |hugging_face| |license| |preprint|

A Modular and Extensible Deep Learning Toolkit for Computer Audition Tasks.

`autrainer` is built on top of `PyTorch <https://pytorch.org/>`_ and `Hydra <https://hydra.cc/>`_,
offering a modular and extensible way to perform reproducible deep learning experiments
for computer audition tasks using YAML configuration files and the command line.


.. _installation:

Installation
------------

It is recommended to install `autrainer` within a virtual environment.
To create a new virtual environment, refer to the `Python venv documentation <https://docs.python.org/3/library/venv.html>`_.

Next, install `autrainer` using `pip`.

.. code-block:: pip

   pip install autrainer

.. note::

   If you are using Windows, make sure that PyTorch (along with ``torchvision`` and ``torchaudio``) version 2.0 or higher is installed beforehand.
   For installation instructions, refer to the `PyTorch documentation <https://pytorch.org/get-started/locally/>`_.


The following optional dependencies can be installed to enable additional features:

* ``latex`` for LaTeX plotting (requires a LaTeX installation).
* ``mlflow`` for `MLflow <https://mlflow.org/>`_ logging.
* ``tensorboard`` for `TensorBoard <https://www.tensorflow.org/tensorboard>`_ logging.
* ``opensmile`` for audio feature extraction with `openSMILE <https://audeering.com/opensmile/>`_.
* ``albumentations`` for image augmentations with `Albumentations <https://albumentations.ai/>`_.
* ``audiomentations`` for audio augmentations with `audiomentations <https://github.com/iver56/audiomentations>`_.
* ``torch-audiomentations`` for audio augmentations with `torch-audiomentations <https://github.com/asteroid-team/torch-audiomentations>`_.

.. code-block:: pip

   pip install autrainer[latex]
   pip install autrainer[mlflow]
   pip install autrainer[tensorboard]
   pip install autrainer[opensmile]
   pip install autrainer[albumentations]
   pip install autrainer[audiomentations]
   pip install autrainer[torch-audiomentations]

To install `autrainer` with all optional dependencies, use the following command:

.. code-block:: pip

   pip install autrainer[all]


To install `autrainer` from source, refer to the `contribution guide <https://autrainer.github.io/autrainer/development/contributing>`_.


Next Steps
----------

To get started using `autrainer`, the `quickstart guide <https://autrainer.github.io/autrainer/usage/quickstart>`_
outlines the creation of a simple training configuration and `tutorials <https://autrainer.github.io/autrainer/usage/tutorials>`_
provide examples for implementing custom modules including their configurations.

For a complete list of available CLI commands, refer to the
`CLI reference <https://autrainer.github.io/autrainer/usage/cli_reference>`_ or the
`CLI wrapper <https://autrainer.github.io/autrainer/usage/cli_wrapper>`_.

.. |pypi| image:: https://img.shields.io/pypi/v/autrainer?logo=pypi&logoColor=b4befe&color=b4befe
   :target: https://pypi.org/project/autrainer/
   :alt: autrainer PyPI Version

.. |python_versions| image:: https://img.shields.io/pypi/pyversions/autrainer?logo=python&logoColor=b4befe&color=b4befe
   :target: https://pypi.org/project/autrainer/
   :alt: autrainer Python Versions

.. |hugging_face| image:: https://img.shields.io/badge/Hugging_Face-autrainer-b4befe?logo=huggingface&logoColor=b4befe
   :target: https://huggingface.co/autrainer
   :alt: autrainer Hugging Face

.. |license| image:: https://img.shields.io/badge/license-MIT-b4befe?logo=c
   :target: https://github.com/autrainer/autrainer/blob/main/LICENSE
   :alt: autrainer GitHub License

.. |preprint| image:: https://img.shields.io/badge/arXiv-2412.11943-AD1C18?logoColor=b4befe&color=b4befe
   :target: https://arxiv.org/abs/2412.11943
   :alt: autrainer arXiv preprint

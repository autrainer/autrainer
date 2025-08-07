.. _contributing:

Contributing
============

Contributions are welcome!
If you would like to contribute to `autrainer`, please open an issue or a pull request on the `GitHub repository <https://github.com/autrainer/autrainer>`_.

Installation
------------

It is recommended to install `autrainer` within a virtual environment.
To create a new virtual environment, refer to the `Python venv documentation <https://docs.python.org/3/library/venv.html>`_.

To set up `autrainer` for development, start by cloning the repository
and then install the development dependencies with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: extended

   git clone git@github.com:autrainer/autrainer.git
   cd autrainer
   uv sync --all-extras --all-groups --inexact


.. note::

   If you are using Windows, make sure that PyTorch (along with :attr:`torchvision` and :attr:`torchaudio`) version 2.0 or higher is installed beforehand.
   For installation instructions, refer to the `PyTorch documentation <https://pytorch.org/get-started/locally/>`_.

   The :attr:`--inexact` flag allows for manual installation of a specific PyTorch version prior to syncing dependencies.
   This is necessary because `autrainer` does not pin a PyTorch version for Windows to not conflict with different CUDA versions.


Conventions
-----------

We use `Ruff <https://docs.astral.sh/ruff/>`_ to enforce code style and formatting.
Common spelling errors are automatically corrected by `codespell <https://github.com/codespell-project/codespell>`_.

Linting, formatting, and spelling checks are run with `pre-commit <https://pre-commit.com/>`_.
To install the pre-commit hooks, run the following command:

.. code-block:: extended

   pre-commit install

Tests
-----

We use `pytest <https://docs.pytest.org/en/stable/>`_ for testing.
To run the tests, use the following command:

.. code-block:: extended

   pytest
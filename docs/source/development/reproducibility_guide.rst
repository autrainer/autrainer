Reproducibility Guide
=====================

To ensure reproducibility of your results, the following guidelines should be followed.
The goal is to allow others to reproduce the results of the project on any system by providing all necessary configurations and source code.

Project Structure
-----------------
The project should be structured in a way that makes it easy to publish and reproduce the results.
Therefore the following project structure is recommended:

* :attr:`conf/`: Directory for storing configuration files.
* :attr:`src/`: Directory for storing additional source code. Additional source code can also be stored in the root directory of the project.
* :file:`requirements.txt`: File containing the pip requirements.
* :attr:`data/`: Directory for storing data files (optional).
* :attr:`results/`: Directory for storing results (optional).

To get started and create the initial project structure, the following :ref:`configuration management <cli_configuration_management>`
CLI command can be used:

.. code-block:: autrainer

   autrainer create --empty

Alternatively, use the following :ref:`configuration management <cli_wrapper_configuration_management>` CLI wrapper function:

.. code-block:: python

   autrainer.cli.create(empty=True)

This will create an empty project structure only including the :file:`conf/config.yaml` configuration file with default values.


Configuration Files
-------------------

All configuration files (not provided by `autrainer` or overridden) should be stored in the :attr:`conf/`
directory and published alongside the :file:`conf/config.yaml` file.
To ensure reproducibility on any system, it is recommended to solely use relative paths in the configuration files.

Custom configuration files (e.g., custom models or datasets) should be placed in the :attr:`conf/` directory.
The implementation of these configurations or further processing scripts should be placed in the :attr:`src/`
directory or in the root directory of the project.

In addition to configurations, the pip requirements should be stored in a :file:`requirements.txt` file in the root directory of the project.

.. code-block:: pip

   pip freeze > requirements.txt

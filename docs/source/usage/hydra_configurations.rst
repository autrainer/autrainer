.. _hydra_configurations:

Hydra Configurations
====================

`autrainer` uses `Hydra <https://hydra.cc/docs/intro/>`_ to configure training experiments.
All configurations are stored in the :file:`conf/` directory and are defined as YAML files.


.. _main_configuration:

Main Configuration
------------------

The main entry point of `autrainer` is the :file:`conf/config.yaml` file by default.
This file defines the configuration of the training experiments over which a grid search is performed.

.. configurations::
   :configs: config
   :exact:

The main configuration file defines the following parameters:

* :attr:`defaults`: A list of default configurations (see :ref:`defaults_list`, :ref:`optional_defaults_list`, and :ref:`autrainer_defaults`).
* :attr:`results_dir`: The directory where the results of the training experiments are stored.
* :attr:`experiment_id`: The ID of the experiment.
* :attr:`hydra/sweeper/params`: The parameters of the sweep.

  * :attr:`\<attr\>`: An attribute to sweep over from the :ref:`defaults_list` (with comma-separated values, e.g., a list of models).
  * :attr:`+\<attr\>`: An attribute to sweep over that is not in the :ref:`defaults_list` (with comma-separated values, e.g., a list of batch sizes).
* :attr:`hydra/sweeper/filters`: A list of filters to filter out unwanted hyperparameter combinations (see :ref:`hydra_sweeper_plugins`).


Some parameters of the main configuration file are outsourced to the :ref:`_autrainer_.yaml defaults <autrainer_defaults>`
file in order to simplify the configuration.

For more information on configuring Hydra, see the `Hydra documentation <https://hydra.cc/docs/patterns/configuring_experiments/>`_.

.. tip::
   To use a different configuration file as the entry point for training experiments,
   use the `-cn/--config-name` argument for the :ref:`autrainer train <cli_training>` CLI command:

   .. code-block:: autrainer
   
      autrainer train -cn some_other_config.yaml

   Alternatively, use the :attr:`config_name` parameter for the :meth:`~autrainer.cli.train` CLI wrapper function (without the file extension):

   .. code-block:: python

      autrainer.cli.train(config_name="some_other_config")
   

   For more information on command line flags, see
   `Hydra's command line flags documentation <https://hydra.cc/docs/advanced/hydra-command-line-flags/>`_.


.. _configuration_directories:

Configuration Directories
-------------------------

Configuration files imported through the :ref:`defaults_list` are stored in the :file:`conf/` directory.
The configuration files are organized in subdirectories (e.g., :file:`conf/dataset/`, :file:`conf/model/`, :file:`conf/optimizer/`, etc.).
This directory structure tells Hydra where to look for the configuration files.
The following configuration subdirectories are available:

* :file:`conf/augmentation/`
* :file:`conf/dataset/`
* :file:`conf/model/`
* :file:`conf/optimizer/`
* :file:`conf/plotting`
* :file:`conf/preprocessing/`
* :file:`conf/scheduler/`


.. _creating_configurations:

Creating Configurations
-----------------------

`autrainer` provides a number of default configurations for models, datasets, optimizers, etc.
that can be used out of the box without creating custom configurations.
To use a default configuration e.g., a `MobileNetV3-Large-T` model, add it to the :file:`conf/config.yaml` file:

.. code-block:: yaml
   :linenos:
   :caption: conf/config.yaml

   ...
   hydra
     sweeper:
       params:
         ...
         model: MobileNetV3-Large-T


.. tip::
   To discover configurations that are available by default, use the :ref:`autrainer list <cli_autrainer_list>` CLI command
   or the :meth:`~autrainer.cli.list` CLI wrapper function.
   For example, to discover all available `MobileNet` configurations, use the following command with a glob pattern:

   .. code-block:: autrainer

       autrainer list model --pattern="MobileNet*"

   Alternatively, use the :meth:`~autrainer.cli.list` CLI wrapper function with the :attr:`pattern` parameter:

   .. code-block:: python

      autrainer.cli.list(directory="model", pattern="MobileNet*")

To create a new configuration, create a new YAML file in the appropriate configuration subdirectory.
Every configuration file should have:

* A unique file name.
* A unique :attr:`id` attribute that matches the file name.
* A :attr:`_target_` attribute that specifies the python import path of the class to be instantiated.
* Optional attributes that are passed to the class constructor as keyword arguments.

For example, to create a new model configuration, create a new YAML file in the :file:`conf/model/` directory:

.. configurations::
   :subdir: model
   :configs: MobileNetV3-Large-T
   :exact:

For more information on how to create custom models, see :ref:`models`.

.. tip::
   To modify an existing configuration provided by `autrainer`, create a new YAML file with the name in the subdirectory.
   The new configuration will override the existing configuration.

   To easily override a configuration, save a local copy of it using the :ref:`autrainer show <cli_autrainer_show>` command:

   .. code-block:: autrainer
   
      autrainer show model "MobileNetV3-Large-T" --save

   Alternatively, use the :meth:`~autrainer.cli.show` CLI wrapper function with the :attr:`save` parameter:

   .. code-block:: python

      autrainer.cli.show(directory="model", id="MobileNetV3-Large-T", save=True)


.. _shorthand_syntax:

Shorthand Syntax
----------------

For configurations that are not primitive types (e.g., numbers or strings) and are not included in the :ref:`defaults_list`,
shorthand syntax is used to reduce the number of configuration files required.
Instead of a configuration file, shorthand syntax configurations can be either a string or a dictionary.

* If the configurations is a string, it is interpreted as the python import path of the class to be instantiated.
* If the configuration is a dictionary, the key is interpreted as the python import path of the class to be instantiated,
  and its values are passed as keyword arguments to the class constructor.


For example, in a dataset configuration, the :attr:`tracking_metric` and the list of :attr:`metrics` are specified with shorthand syntax:

.. literalinclude:: ../examples/metrics.yaml
    :language: yaml
    :caption: conf/dataset/ExampleDataset.yaml
    :linenos:

Shorthand syntax is used to specify the pipeline of :ref:`augmentations`, the :ref:`loggers`, the :ref:`metrics` for the dataset,
and the model as well as :ref:`transforms`.


.. _interpolation_syntax:

Interpolation Syntax
--------------------

`autrainer` supports `OmegaConf <https://omegaconf.readthedocs.io/en/2.3_branch/index.html>`_
variable interpolation to reference attributes from anywhere in the configuration.

For example, the :ref:`custom loggers tutorial <tut_loggers>` uses the OmegaConf interpolation syntax to reference the :attr:`results_dir`
from the :ref:`main configuration <main_configuration>` file to set the output directory of the :class:`WandBLogger`:

.. code-block:: yaml
   :caption: conf/config.yaml
   :linenos:

   ...
   loggers:
     - wandb_logger.WandBLogger:
         output_dir: ${results_dir}/.wandb
   ...

For more information on variable interpolation, refer to the
`OmegaConf documentation <https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation>`_.


.. _defaults_list:

Defaults List
-------------

The defaults list instructs Hydra to import configurations from other YAML files to build the final configuration.
For more information on the defaults list, see `Hydra's defaults list documentation <https://hydra.cc/docs/advanced/defaults_list/>`_.

For brevity, the defaults list is outsourced to :ref:`_autrainer_.yaml defaults <autrainer_defaults>`
file with the following imports:

.. code-block:: yaml
   :linenos:
   :caption: \_autrainer\_.yaml

   defaults:
     - _self_
     - dataset: ??? # placeholder
     - model: ??? # placeholder
     - optimizer: ??? # placeholder
     - scheduler: None # optional default
     - augmentation: None # optional default
     - plotting: Default # optional default


.. _optional_defaults_list:

Optional Defaults and Overrides
-------------------------------

Optional defaults like :attr:`scheduler` and :attr:`augmentation` are set to :attr:`None`
by default and not required to be defined in the main configuration.
:attr:`None` is a special value that tells Hydra to ignore the default and e.g., not use a :ref:`scheduler <schedulers>`
or :ref:`augmentation <augmentations>` for training.

Optional defaults which are not overridden in the :attr:`hydra/sweeper/params` configuration, can be overridden using the :attr:`override`
keyword in the defaults list of the :ref:`main configuration <main_configuration>` file.
This includes the :ref:`plotting <core_plotting>` configuration or Hydra plugins like :ref:`sweepers <hydra_sweeper_plugins>`
and :ref:`launchers <hydra_launcher_plugins>`.

For more information on overriding defaults, refer to the
`Hydra overriding documentation <https://hydra.cc/docs/advanced/defaults_list/#overriding-config-group-options>`_.


.. _autrainer_defaults:

autrainer Defaults
----------------------

The :file:`_autrainer_.yaml` file contains further default configurations to simplify the :ref:`main configuration <main_configuration>`,
which includes:

* The :ref:`defaults list <defaults_list>` and :ref:`optional defaults list <optional_defaults_list>`.
* Global default parameters for :ref:`training <training>`, such as the evaluation frequency, save frequency, inference batch size, CUDA-enabled device, etc.
* Hydra configurations for always starting a `Hydra multirun <https://hydra.cc/docs/1.0/tutorials/basic/running_your_app/multi-run/>`_ (grid search)
  and setting the output directory and experiment name according to the current configuration.

.. tip::

   Any global default parameter can be overridden in the :ref:`main configuration <main_configuration>` file by redefining it.

.. configurations::
   :configs: _autrainer_
   :exact:


.. _hydra_plugins:

Hydra Plugins
--------------

Any Hydra :ref:`sweeper <hydra_sweeper_plugins>` or :ref:`launcher <hydra_launcher_plugins>` plugin can be used to customize
the hyperparameter search or parallelize the training jobs.

.. tip::
   
   Sweepers and launchers can be combined to perform more complex hyperparameter optimizations and parallelize training jobs.


.. _hydra_sweeper_plugins:

Sweeper Plugins
~~~~~~~~~~~~~~~

By default, `autrainer` uses the `hydra-filter-sweeper <https://github.com/autrainer/hydra-filter-sweeper>`_
plugin to sweep over hyperparameter configurations.
This plugin allows to specify a list of :attr:`filters` in the configuration file to filter out unwanted hyperparameter combinations.

.. tip::

   To specify custom :attr:`filters`, refer to the :ref:`filtering out configurations quickstart guide <quick_grid_filters>` and the
   `hydra-filter-sweeper documentation <https://github.com/autrainer/hydra-filter-sweeper>`_.


.. note::

   If no :attr:`filters` are specified, the plugin will not filter out any configurations and resemble the behavior of the default
   `Hydra basic sweeper <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/#sweeper>`_ plugin.

To perform more complex hyperparameter sweeps, different sweeper plugins can be used.

For example, the `Hydra Optuna Sweeper plugin <https://hydra.cc/docs/plugins/optuna_sweeper/>`_
can be used to perform hyperparameter optimization using `Optuna <https://optuna.org/>`_.

To install the Optuna Sweeper plugin, run the following command:

.. code-block:: pip

   pip install hydra-optuna-sweeper

The following configuration uses the Optuna Sweeper plugin to perform 10 trials with different learning rates:

.. literalinclude:: ../examples/config_optuna.yaml
   :language: yaml
   :caption: conf/config.yaml
   :linenos:

.. note::
   `autrainer` returns the best validation value of the dataset :ref:`tracking metric <metrics>` as the objective for optimization.
   The optimization :attr:`direction` attribute in the configuration file can be set to `minimize` or `maximize`
   according to the dataset tracking metric.


.. _hydra_launcher_plugins:

Launcher Plugins
~~~~~~~~~~~~~~~~

By default, `autrainer` uses the Hydra basic launcher to sequentially launch the jobs defined in the configuration file.

To parallelize the training jobs, different launcher plugins can be used.

For example, the `Hydra Submitit Launcher plugin <https://hydra.cc/docs/plugins/submitit_launcher/>`_
can be used to parallelize the training jobs using `Submitit <https://github.com/facebookincubator/submitit>`_.

To install the Submitit Launcher plugin, run the following command:

.. code-block:: pip

   pip install hydra-submitit-launcher

The following configuration uses the Submitit Launcher plugin to parallelize the training jobs:

.. literalinclude:: ../examples/config_submitit.yaml
   :language: yaml
   :caption: conf/config.yaml
   :linenos:
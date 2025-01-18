.. _quickstart:

Quickstart
==========

The following quickstart guide provides a short introduction to `autrainer` and the creation of simple training experiments.


.. _quick_first_experiment:

First Experiment
----------------

To get started, create a new directory and navigate to it:

.. code-block:: extended

    mkdir autrainer_example && cd autrainer_example

Next, create a new empty `autrainer` project using the following :ref:`configuration management <cli_configuration_management>` CLI command:

.. code-block:: autrainer

    autrainer create --empty

Alternatively, use the following :ref:`configuration management <cli_wrapper_configuration_management>` CLI wrapper function:

.. code-block:: python

    import autrainer.cli # the import is omitted in the following examples for brevity

    autrainer.cli.create(empty=True)


This will create the :ref:`configuration directory structure <configuration_directories>`
and the :ref:`main configuration <main_configuration>` (:file:`conf/config.yaml`) file with default values:

.. configurations::
   :configs: config
   :exact:


Now, run the following :ref:`training <cli_training>` command to train the model:

.. code-block:: autrainer

    autrainer train

Alternatively, use the following :ref:`training <cli_wrapper_training>` CLI wrapper function:

.. code-block:: python

    autrainer.cli.train() # the train function is omitted in the following examples for brevity

This will train the default :attr:`ToyFFNN` feed-forward neural network (:class:`~autrainer.models.FFNN`) on the default
:attr:`ToyTabular-C` classification dataset with tabular data (:class:`~autrainer.datasets.ToyDataset`)
and output the training results to the :file:`results/default/` directory.


Custom Model Configuration
--------------------------

The :ref:`first experiment <quick_first_experiment>` uses the default :attr:`ToyFFNN` model with the following configuration having 2 hidden layers:

.. configurations::
   :subdir: model
   :configs: ToyFFNN
   :exact:

To :ref:`create another configuration <creating_configurations>` for the :class:`~autrainer.models.FFNN` model with 3 hidden layers,
create a new configuration file in the :file:`conf/model/` directory:

.. literalinclude:: ../examples/quickstart/model_hidden_layers.yaml
   :language: yaml
   :caption: conf/model/Three-Layer-FFNN.yaml
   :linenos:

Next, update the :ref:`main configuration <main_configuration>` (:file:`conf/config.yaml`) file to use the new model configuration:

.. literalinclude:: ../examples/quickstart/config_hidden_layers.yaml
   :language: yaml
   :caption: conf/config.yaml
   :linenos:

Now, run the following :ref:`training <cli_training>` command to train the model with 3 hidden layers:

.. code-block:: autrainer

    autrainer train


.. _quick_grid_search:

Grid Search Configuration
-------------------------

To perform a grid search over multiple multiple configurations defined in the :attr:`params`, update the
:ref:`main configuration <main_configuration>` (:file:`conf/config.yaml`) to include multiple values separated by a comma.

The following configuration performs a grid search over the default :class:`~autrainer.models.FFNN` model with 2 and 3 hidden layers
as well as 3 different seeds:

.. literalinclude:: ../examples/quickstart/config_grid_search.yaml
   :language: yaml
   :caption: conf/config.yaml
   :linenos:


Now, run the following :ref:`training <cli_training>` command to train the models with 2 and 3 hidden layers and 3 different seeds:

.. code-block:: autrainer

    autrainer train

By default, a grid search is performed sequentially.
`Hydra <https://hydra.cc/>`_ allows the use of different :ref:`launcher plugins <hydra_launcher_plugins>` to perform parallel grid searches.

.. note::
  
   If a run already exists in the same experiment and has been completed successfully, then it will be skipped.
   This may be the case for both the default and custom model configurations with seed 1
   if they have already been trained in the previous examples.

To compare the results of the individual runs as well as averaged across seeds, run the following :ref:`postprocessing <cli_postprocessing>` command:

.. code-block:: autrainer

    autrainer postprocess results default --aggregate seed

Alternatively, use the following :ref:`postprocessing <cli_wrapper_postprocessing>` CLI wrapper function:

.. code-block:: python

    autrainer.cli.postprocess(
        results_dir="results",
        experiment_id="default",
        aggregate=[["seed"]],
    )

.. _quick_audio_classification:

Spectrogram Classification
--------------------------

To train a :class:`~autrainer.models.Cnn10` model on an :ref:`audio dataset <datasets>` such as :class:`~autrainer.datasets.DCASE2016Task1`,
update the :ref:`main configuration <main_configuration>` (:file:`conf/config.yaml`) file:

.. literalinclude:: ../examples/quickstart/config_audio.yaml
   :language: yaml
   :caption: conf/config.yaml
   :linenos:

For the :class:`~autrainer.models.Cnn10` model, the following configuration is used:

.. configurations::
   :subdir: model
   :configs: Cnn10-32k-T
   :exact:

The ending :attr:`32k-T` indicates that the model using transfer learning and has been pretrained with a sample rate of 32 kHz.

.. tip::
   
   To discover all available default configurations for e.g., different models,
   the :ref:`configuration management CLI <cli_configuration_management>`,
   the :ref:`configuration management CLI wrapper <cli_wrapper_configuration_management>`,
   and the :ref:`models documentation <models>` can be used.

For the :class:`~autrainer.datasets.DCASE2016Task1` dataset, the following configuration is used:

.. configurations::
   :subdir: dataset
   :configs: DCASE2016Task1-32k
   :exact:

The ending :attr:`32k` indicates that the dataset has a sample rate of 32 kHz and provides log-Mel spectrograms instead of raw audio.

To avoid race conditions when using :ref:`hydra_launcher_plugins` that may run multiple training jobs in parallel,
the following :ref:`preprocessing <cli_preprocessing>` command is used to fetch and download the model weights and the raw audio files of the dataset:

.. code-block:: autrainer

    autrainer fetch
   
Alternatively, use the following :ref:`preprocessing <cli_wrapper_preprocessing>` CLI wrapper function:

.. code-block:: python

    autrainer.cli.fetch()

As the dataset uses log-Mel spectrograms instead of the raw audio files downloaded in the previous step,
the following :ref:`preprocessing <cli_preprocessing>` command is used to preprocess and extract the features from the raw audio files:

.. code-block:: autrainer

    autrainer preprocess

Alternatively, use the following :ref:`preprocessing <cli_wrapper_preprocessing>` CLI wrapper function:

.. code-block:: python

    autrainer.cli.preprocess()

Now, run the following :ref:`training <cli_training>` command to train the model on the audio dataset:

.. code-block:: autrainer

    autrainer train


.. _quick_training_configurations:

Training Duration & Step-based Training
---------------------------------------

By default, `autrainer` uses epoch-based :ref:`training <training>`, where the :attr:`iterations` correspond to the number of epochs.
To change the training duration of the :ref:`spectrogram classification model <quick_audio_classification>`,
increase the number of :attr:`iterations` in the :ref:`main configuration <main_configuration>` (:file:`conf/config.yaml`) file.

However, to use step-based training instead of epoch-based training, set the :attr:`training_type` to :attr:`step`.

The following configuration trains the :ref:`spectrogram classification model <quick_audio_classification>`
for a total of 1000 steps with step-based training, evaluating every 100 steps,
saving the states every 200 steps, and without displaying a progress bar:

.. literalinclude:: ../examples/quickstart/config_step.yaml
   :language: yaml
   :caption: conf/config.yaml
   :linenos:

Now, run the following :ref:`training <cli_training>` command to train the model on the audio dataset for 1000 steps:

.. code-block:: autrainer

    autrainer train


.. _quick_grid_filters:

Filtering Configurations
----------------------------

By default, `autrainer` filters out any configurations that have already been trained
and exist in the same experiment using the `hydra-filter-sweeper <https://github.com/autrainer/hydra-filter-sweeper/>`_ plugin
with the following :attr:`filters` that are implicitly set in the :ref:`_autrainer_.yaml defaults <autrainer_defaults>` file:

.. literalinclude:: ../examples/quickstart/config_filters.yaml
   :language: yaml
   :caption: conf/config.yaml
   :linenos:
   :lines: 18-20


To filter out unwanted configurations and exclude them from training,
the `hydra-filter-sweeper <https://github.com/autrainer/hydra-filter-sweeper/>`_ plugin can be used as the 
:ref:`Hydra sweeper plugin <hydra_sweeper_plugins>`.
:attr:`hydra-filter-sweeper` allows to specify a list of :attr:`filters` to exclude configurations based on their attributes.

The following configuration expands the :ref:`grid search <quick_grid_search>` configuration
and adds a filter that excludes any seed greater than 2 for the :class:`Three-Layer-FFNN` model:

.. literalinclude:: ../examples/quickstart/config_filters.yaml
   :language: yaml
   :caption: conf/config.yaml
   :linenos:

.. note::

   If the :attr:`filters` attribute is overridden in the :ref:`main configuration <main_configuration>` (:file:`conf/config.yaml`) file,
   then the default filters are not applied.
   To still filter out configurations that have already been trained, the following default filter should still be included:
   
   .. literalinclude:: ../examples/quickstart/config_filters.yaml
      :language: yaml
      :caption: conf/config.yaml
      :linenos:
      :lines: 18-20

Now, run the following :ref:`training <cli_training>` command to train the :class:`ToyFFNN` with 3 seeds and the :class:`Three-Layer-FFNN` with 2 seeds:

.. code-block:: autrainer

    autrainer train


Next Steps
----------

For more information on creating configurations, refer to the :ref:`Hydra configurations <hydra_configurations>`
as well as the `Hydra <https://hydra.cc/>`_ documentation.

To create custom implementations alongside configurations, refer to the :ref:`tutorials <tutorials>`.
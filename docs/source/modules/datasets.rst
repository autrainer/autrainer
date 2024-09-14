.. _datasets:

Datasets
========

`autrainer` provides a number of different audio-specific datasets, base datasets for different tasks, and toy datasets for testing purposes.

.. tip::

   To create custom datasets, refer to the :ref:`custom datasets tutorial <tut_datasets>`.


A dataset configuration file should include the following attributes, in addition to the common attributes like :attr:`id`, :attr:`_target_`,
and dataset-specific attributes inferred from the constructors:

**Structure and Loading**

* :attr:`path`: The directory path to the dataset, containing the :attr:`features_subdir` directory
  and the :file:`train.csv`, :file:`dev.csv`, and :file:`test.csv` files.
* :attr:`features_subdir`: The subdirectory within the dataset directory where (extracted) features are stored.
  If different from `default` and set to the name of a :ref:`preprocessing transform <preprocessing_transforms>`,
  :ref:`autrainer preprocess <cli_preprocessing>` will save the processed data in the specified subdirectory with the same name.
* :attr:`index_column`: The column in the CSV files that contains paths to the audio files,
  relative to the dataset subdirectory.
* :attr:`target_column`: The column in the CSV files that contains the targets or labels.
* :attr:`file_type`: The type of the files to load (e.g., `wav`, `npy`).
* :attr:`file_handler`: The :ref:`file handler <dataset_file_handlers>` to use for loading the data.

**Training and Evaluation**

* :attr:`criterion`: The :ref:`criterion <criterions>` to use for training.
* :attr:`metrics`: A list of :ref:`metrics <metrics>` to evaluate the model.
* :attr:`tracking_metric`: The :ref:`metric <metrics>` to track for early stopping and model selection.
* :attr:`transform`: The :ref:`online transforms <online_transforms>` to apply to the data and the output :attr:`type` of the dataset.


To avoid race conditions when using :ref:`hydra_launcher_plugins` that may run multiple training jobs in parallel,
:ref:`autrainer fetch <cli_autrainer_fetch>` and :ref:`autrainer preprocess <cli_preprocessing>`
or :meth:`~autrainer.cli.fetch` and :meth:`~autrainer.cli.preprocess`
are used to download the dataset and preprocess the data before training.


.. note::

   All datasets that are provided by `autrainer` can be automatically downloaded as well as optionally preprocessed using the
   :ref:`autrainer fetch <cli_autrainer_fetch>` and :ref:`autrainer preprocess <cli_preprocessing>` CLI commands or the
   :meth:`~autrainer.cli.fetch` and :meth:`~autrainer.cli.preprocess` CLI wrapper functions.



Abstract Dataset
------------------

All datasets inherit from the :class:`AbstractDataset` class.

.. autoclass:: autrainer.datasets.AbstractDataset
   :members:

Base Datasets
------------------

Base datasets that can be used for training without the need for creating custom datasets.

.. autoclass:: autrainer.datasets.BaseClassificationDataset
   :members:

.. autoclass:: autrainer.datasets.BaseMLClassificationDataset
   :members:

.. autoclass:: autrainer.datasets.BaseRegressionDataset
   :members:


Toy Datasets
------------------

A toy dataset for testing purposes.

.. note::

   To easily test implementations, multiple toy dataset configurations across modalities and tasks are provided.
   We offer :attr:`ToyAudio-...` for audio, :attr:`ToyImage-...` for image, and :attr:`ToyTabular-...` for tabular data, respectively.
   For each dataset, we provide a task :attr:`...-R` for regression, :attr:`...-C` for classification and :attr:`...-MLC` for multi-label classification.

.. autoclass:: autrainer.datasets.ToyDataset

 .. dropdown:: Default Configurations

    .. configurations::
       :subdir: dataset
       :configs: ToyAudio ToyImage ToyTabular
       :headline:


.. _audio_datasets:

Audio Datasets
------------------

We provide a number of different audio-specific datasets.

.. autoclass:: autrainer.datasets.AIBO
   :members:
   
   .. dropdown:: Default Configurations
  
      .. configurations::
         :subdir: dataset
         :configs: AIBO
  
.. autoclass:: autrainer.datasets.DCASE2016Task1
   :members:

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: dataset
         :configs: DCASE2016Task1

.. autoclass:: autrainer.datasets.DCASE2018Task3
   :members:

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: dataset
         :configs: DCASE2018Task3

.. autoclass:: autrainer.datasets.DCASE2020Task1A
   :members:

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: dataset
         :configs: DCASE2020Task1A

.. autoclass:: autrainer.datasets.EDANSA2019
   :members:

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: dataset
         :configs: EDANSA2019

.. autoclass:: autrainer.datasets.EmoDB
   :members:

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: dataset
         :configs: EmoDB

.. autoclass:: autrainer.datasets.SpeechCommands
   :members:

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: dataset
         :configs: SpeechCommands

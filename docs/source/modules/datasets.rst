.. _datasets:

Datasets
========

`autrainer` provides a number of different audio-specific datasets, base datasets for different tasks, and toy datasets for testing purposes.
To ensure consistency across different data formats and manage multiple data types, all datasets should follow a standardized structure.

.. tip::

   To create custom datasets, refer to the :ref:`custom datasets tutorial <tut_datasets>`.


In addition to the common attributes like :attr:`id`, :attr:`_target_`, the dataset configuration file should include the following attributes:

**Structure and Loading**

* :attr:`path`: Directory path containing the :attr:`features_subdir` directory and corresponding CSV files
  (such as :file:`train.csv`, :file:`dev.csv`, and :file:`test.csv`).
* :attr:`features_subdir`: The subdirectory within the dataset path where (extracted) features are stored.

  * If no preprocessing is used (e.g., for raw audio), it should be :attr:`default`.
  * For :ref:`preprocessing transforms <preprocessing_transforms>` (e.g., log-Mel spectrograms with :attr:`log_mel_16k`),
    it should match the transform's name, and the processed features are saved in this subdirectory after preprocessing.

* :attr:`index_column`: Column in the CSV files containing the file paths, relative to the :attr:`features_subdir` directory.
* :attr:`target_column`: Column in the CSV files containing the corresponding targets or labels for each file.
* :attr:`file_type`: Specifies the type of files to be loaded (e.g., :attr:`wav`, :attr:`npy`, etc.).
* :attr:`file_handler`: The :ref:`file handler <dataset_file_handlers>` used for loading the files.

This results in a directory structure like the following:

.. code-block:: python

   {path}/{features_subdir}/optional/subdirs/some.file


For instance, a file in the :attr:`index_column` might be :file:`optional/subdirs/some.file`,
where :file:`some.file` is an audio or a feature file.

In order to load custom dataset splits that do not follow the standard :file:`train.csv`, :file:`dev.csv`, and :file:`test.csv` convention,
the :attr:`~autrainer.datasets.AbstractDataset.df_train`, :attr:`~autrainer.datasets.AbstractDataset.df_dev`,
and :attr:`~autrainer.datasets.AbstractDataset.df_test`, properties of the dataset class can be overwritten
(see :ref:`custom datasets tutorial <tut_datasets>`).

**Training and Evaluation**

* :attr:`criterion`: The :ref:`criterion <criterions>` to use for training.
* :attr:`metrics`: A list of :ref:`metrics <metrics>` to evaluate the model.
* :attr:`tracking_metric`: The :ref:`metric <metrics>` to track for early stopping and model selection.
* :attr:`transform`: The :ref:`online transforms <online_transforms>` to apply to the data and the output :attr:`type` of the dataset.
* :attr:`train_loader_kwargs`, :attr:`dev_loader_kwargs`, and :attr:`test_loader_kwargs`:
  Additional keyword arguments for the :class:`~torch.utils.data.DataLoader` such as the :attr:`num_workers`, :attr:`prefetch_factor`, etc.
  The keyword arguments can also be specified globally in the :ref:`main configuration <main_configuration>` file, which will be passed to all datasets.
  However, the dataset-specific keyword arguments will overwrite the global ones.


.. note::

   The following attributes are automatically passed to the dataset during initialization and determined at runtime:
   
   * :attr:`train_transform`, :attr:`dev_transform`, and :attr:`test_transform`: The :class:`~autrainer.transforms.SmartCompose`
     transformation pipelines (which may include possible :ref:`online transforms <online_transforms>` or :ref:`augmentations <augmentations>`).
   * :attr:`seed`: The random seed for reproducibility during training.

   The :attr:`transform` attribute in the configuration is not passed to the dataset during initialization
   and is used to specify the :ref:`type of data <online_transforms>` the dataset provides as well as any
   :ref:`online transforms <online_transforms>` to be applied to the data at runtime.


To avoid race conditions when using :ref:`hydra_launcher_plugins` that may run multiple training jobs in parallel,
:ref:`autrainer fetch <cli_autrainer_fetch>` and :ref:`autrainer preprocess <cli_preprocessing>`
or :meth:`~autrainer.cli.fetch` and :meth:`~autrainer.cli.preprocess`
are used to download the dataset and preprocess the data before training.

.. note::

   All datasets that are provided by `autrainer` can be automatically downloaded as well as optionally preprocessed using the
   :ref:`autrainer fetch <cli_autrainer_fetch>` and :ref:`autrainer preprocess <cli_preprocessing>` CLI commands or the
   :meth:`~autrainer.cli.fetch` and :meth:`~autrainer.cli.preprocess` CLI wrapper functions.


All `autrainer` datasets and loaders return :ref:`DataItem and DataBatch structs <core_data_items>` which represent the data items
and batches of data, respectively.


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

.. autoclass:: autrainer.datasets.BaseMTRegressionDataset
   :members:


Toy Datasets
------------------

A toy dataset for testing purposes.

.. note::

   To easily test implementations, multiple toy dataset configurations across modalities and tasks are provided.
   We offer :attr:`ToyAudio-...` for audio, :attr:`ToyImage-...` for image, and :attr:`ToyTabular-...` for tabular data, respectively.
   For each dataset, we provide a task :attr:`...-R` for regression, :attr:`...-C` for classification, :attr:`...-MLC` for multi-label classification,
   and :attr:`...-MTR` for multi-target regression.

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

.. autoclass:: autrainer.datasets.AudioSet
   :members:
   
   .. dropdown:: Default Configurations
  
      .. configurations::
         :subdir: dataset
         :configs: AudioSet
  
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


.. autoclass:: autrainer.datasets.MSPPodcast
   :members:

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: dataset
         :configs: MSPPodcast


.. autoclass:: autrainer.datasets.SpeechCommands
   :members:

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: dataset
         :configs: SpeechCommands

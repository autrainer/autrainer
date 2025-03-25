.. _tutorials:

Tutorials
=========

`autrainer` is designed to be flexible and extensible, allowing for the creation of custom ...

* :ref:`models <tut_models>`
* :ref:`datasets <tut_datasets>` (including :ref:`metrics <tut_metrics>`, :ref:`criterions <tut_criterions>`,
  :ref:`file handlers <tut_file_handlers>`, :ref:`target transforms <tut_target_transforms>`, 
  and :ref:`advanced data pipelines <tut_advanced>`)
* :ref:`optimizers <tut_optimizers>`
* :ref:`schedulers <tut_schedulers>`
* :ref:`transforms <tut_transforms>` (including :ref:`preprocessing transforms <tut_preprocessing_transforms>`
  and :ref:`online transforms <tut_online_transforms>`)
* :ref:`augmentations <tut_augmentations>`
* :ref:`loggers <tut_loggers>`

For each, a tutorial is provided below to demonstrate their implementation and configuration.

For the following tutorials, all python files should be placed in the project root directory
and all configuration files should be placed in the corresponding subdirectories of the :file:`conf/` directory.


.. _tut_models:

Custom Models
-------------

To create a custom :ref:`model <models>`, inherit from :class:`~autrainer.models.AbstractModel`
and implement the :meth:`~autrainer.models.AbstractModel.forward` and :meth:`~autrainer.models.AbstractModel.embeddings` methods.
All arguments of the constructor have to be assigned to a variable with the same name, as :class:`~autrainer.models.AbstractModel`
inherits from `audobject <https://audeering.github.io/audobject/api/audobject.Object.html#audobject.Object>`_.

For example, the following model is a simple CNN that takes a spectrogram as input and has a variable number of hidden CNN layers with
a different number of filters each:

.. literalinclude:: ../examples/tutorials/spectrogram_cnn.py
   :language: python
   :caption: spectrogram_cnn.py
   :linenos:

Next, create a :file:`SpectrogramCNN.yaml` configuration file for the model in the :file:`conf/model/` directory:

.. literalinclude:: ../examples/tutorials/SpectrogramCNN.yaml
   :language: yaml
   :caption: conf/model/SpectrogramCNN.yaml
   :linenos:

The :attr:`id` should match the name of the configuration file.
The :attr:`_target_` should point to the custom model class via a python import path
(here assuming that the :file:`spectrogram_cnn.py` file is in the root directory of the project).
Each model should include a :attr:`transform/type` attribute in the configuration file,
specifying the :ref:`input type <online_transforms>` it expects.

.. note::
   The :attr:`output_dim` attribute is automatically passed to the model during initialization
   and determined by the :ref:`dataset <datasets>` at runtime.

   The :attr:`transform` attribute in the configuration is not passed to the model during initialization
   and is used to specify the :ref:`input type <online_transforms>` of the model and any
   :ref:`online transforms <online_transforms>` to be applied to the data at runtime.
  
.. _tut_datasets:

Custom Datasets
---------------

To create a custom :ref:`dataset <datasets>`, inherit from :class:`~autrainer.datasets.AbstractDataset`
and implement the :attr:`~autrainer.datasets.AbstractDataset.target_transform`
and :attr:`~autrainer.datasets.AbstractDataset.output_dim` properties.

The train, dev, and test datasets as well as loaders are automatically created by the abstract class.
However, this requires that the dataset structure follows the standard format outlined in the :ref:`dataset documentation <datasets>`.
If the dataset structure is different or does not rely on dataframes, the 
:attr:`~autrainer.datasets.AbstractDataset.df_train`, :attr:`~autrainer.datasets.AbstractDataset.df_dev`,
and :attr:`~autrainer.datasets.AbstractDataset.df_test`, :attr:`~autrainer.datasets.AbstractDataset.train_dataset`,
:attr:`~autrainer.datasets.AbstractDataset.train_loader` etc. properties can be overridden.

`autrainer` provides base datasets for classification (:class:`~autrainer.datasets.BaseClassificationDataset`),
regression (:class:`~autrainer.datasets.BaseRegressionDataset`),
and multi-label classification (:class:`~autrainer.datasets.BaseMLClassificationDataset`) tasks.
In this case, both the target transform and output dimension are already implemented in the base class and do not need to be overridden.

.. tip::
   
   To automatically download a custom dataset, implement the :meth:`~autrainer.datasets.AbstractDataset.download` method.
   This method is called by the :ref:`autrainer fetch CLI command <cli_preprocessing>` as well as the :meth:`~autrainer.cli.fetch` CLI wrapper function.
   The :attr:`path` attribute specified in the dataset configuration file is passed to the method to store the downloaded data in.


**ESC-50 Example**

For example, the `ESC-50 <https://github.com/karolpiczak/ESC-50>`_ dataset is an audio classification dataset and can be implemented as follows:

.. literalinclude:: ../examples/tutorials/esc_50.py
   :language: python
   :caption: esc_50.py
   :linenos:

The dataset provides audio files by default (which are moved to the :file:`default/` directory in the
:meth:`~autrainer.datasets.AbstractDataset.download` method) and the corresponding metadata of the dataset is stored in the :file:`esc50.csv` file.

To allow the the specification of custom folds, the :attr:`~autrainer.datasets.AbstractDataset.df_train`,
:attr:`~autrainer.datasets.AbstractDataset.df_dev`, and :attr:`~autrainer.datasets.AbstractDataset.df_test` properties are overridden
to split the :file:`esc50.csv` metadata file into the respective train, dev, and test dataframes.
This also allows for cross-validation by creating multiple configurations with different folds.

To extract log-Mel spectrograms from the audio files, a :ref:`preprocessing transform <preprocessing_transforms>`
can be applied to the data before training.
The following configuration creates a new :file:`ESC50-32k.yaml` dataset in the :file:`conf/dataset/` directory with log-Mel spectrograms
preprocessed at a sample rate of 32 kHz:

.. literalinclude:: ../examples/tutorials/ESC50-32k.yaml
   :language: yaml
   :caption: conf/dataset/ESC50-32k.yaml
   :linenos:

The dataset can be automatically downloaded and preprocessed using the :ref:`autrainer fetch <cli_autrainer_fetch>`
and :ref:`autrainer preprocess <cli_preprocessing>` CLI commands
or the :meth:`~autrainer.cli.fetch` and :meth:`~autrainer.cli.preprocess` CLI wrapper functions.


**Simple Dataset Example**

If the structure of the dataset follows the standard format outlined in the :ref:`dataset documentation <datasets>`,
no implementation is necessary and a new dataset can be created by simply adding a configuration file to the :file:`conf/dataset/` directory.

For example, the following configuration file creates a new :file:`SpectrogramDataset.yaml` classification dataset,
preprocessing the data with a :ref:`spectrogram preprocessing transform <preprocessing_transforms>` at a sample rate of 32 kHz:

.. literalinclude:: ../examples/tutorials/SpectrogramDataset.yaml
   :language: yaml
   :caption: conf/dataset/SpectrogramDataset.yaml
   :linenos:

This dataset assumes that the :file:`data/SpectrogramDataset` directory contains the following directories and files:

* :file:`default/` directory containing the raw audio files.
  These audio files are preprocessed using the :ref:`spectrogram preprocessing transform <preprocessing_transforms>`
  with the :ref:`autrainer preprocess CLI command <cli_preprocessing>` or the :meth:`~autrainer.cli.preprocess` CLI wrapper function
  and stored in the :file:`data/SpectrogramDataset/log_mel_32k` directory.
* :file:`train.csv`, :file:`dev.csv`, and :file:`test.csv` files containing the file paths relative to the :file:`default/` directory
  in the :attr:`index_column` column and the corresponding labels in the :attr:`target_column` column.


.. _tut_metrics:

Custom Metrics
~~~~~~~~~~~~~~

To create a custom :ref:`metric <metrics>`, inherit from :class:`~autrainer.metrics.AbstractMetric`
and implement the :attr:`~autrainer.metrics.AbstractMetric.starting_metric`,
:attr:`~autrainer.metrics.AbstractMetric.suffix` properties,
as well as the :meth:`~autrainer.metrics.AbstractMetric.get_best`,
the :meth:`~autrainer.metrics.AbstractMetric.get_best_pos`,
and :meth:`~autrainer.metrics.AbstractMetric.compare` static methods.

`autrainer` provides base classes for ascending (:class:`~autrainer.metrics.BaseAscendingMetric`) and descending
(:class:`~autrainer.metrics.BaseDescendingMetric`) metrics that can be inherited from to simplify the implementation.

For example, the following metric implements the
`Cohen's Kappa <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html>`_
score with either linear or quadratic weights:

.. literalinclude:: ../examples/tutorials/cohens_kappa_metric.py
   :language: python
   :caption: cohens_kappa_metric.py
   :linenos:

The :attr:`fn` attribute is the function that is automatically called in the :meth:`~autrainer.metrics.AbstractMetric.__call__` method
and the :attr:`weights` attribute is passed to the :attr:`fn` as a keyword argument.

As :ref:`metrics <metrics>` are specified using :ref:`shorthand syntax <shorthand_syntax>` in the dataset configuration,
the following relative import path can be used to reference it as the :attr:`tracking_metric` for the dataset:

.. code-block:: yaml
   :caption: conf/dataset/ExampleDataset.yaml
   :linenos:
   
   ...
   tracking_metric:
     cohens_kappa_metric.CohensKappa:
        weights: linear # linear or quadratic
   ...



.. _tut_criterions:

Custom Criterions
~~~~~~~~~~~~~~~~~

To create a custom :ref:`criterion <criterions>`, inherit from :class:`torch.nn.modules.loss._Loss` and implement the :meth:`forward` method.
If the criterion relies on the dataset, an optional :ref:`criterion setup method <criterion_setup>`
can be defined which is called after the dataset is initialized.

.. note::

   The :attr:`reduction` attribute of each criterion is automatically set to :attr:`"none"` during instantiation and
   the :meth:`forward` method should return the per-example loss.

For example, the following criterion implements `CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_
with an additional scaling factor:

.. literalinclude:: ../examples/tutorials/scaled_ce_loss.py
   :language: python
   :caption: scaled_ce_loss.py
   :linenos:

As :ref:`criterions <criterions>` are specified using :ref:`shorthand syntax <shorthand_syntax>` in the dataset configuration,
the following relative import path can be used to reference it as the :attr:`criterion` for the dataset:

.. code-block:: yaml
   :caption: conf/dataset/ExampleDataset.yaml
   :linenos:
   
   ...
   criterion:
     scaled_ce_loss.ScaledCrossEntropyLoss:
        scaling_factor: 0.5
   ...

Or without overriding the default :attr:`scaling_factor` value:

.. code-block:: yaml
   :caption: conf/dataset/ExampleDataset.yaml
   :linenos:
   
   ...
   criterion: scaled_ce_loss.ScaledCrossEntropyLoss
   ...


.. _tut_file_handlers:

Custom File Handlers
~~~~~~~~~~~~~~~~~~~~

To create a custom :ref:`file handler <dataset_file_handlers>`, inherit from :class:`~autrainer.datasets.utils.AbstractFileHandler`
and implement the :meth:`~autrainer.datasets.utils.AbstractFileHandler.load` and :meth:`~autrainer.datasets.utils.AbstractFileHandler.save` methods.

For example, the following file handler loads and saves PyTorch tensors:

.. literalinclude:: ../examples/tutorials/torch_file_handler.py
   :language: python
   :caption: torch_file_handler.py
   :linenos:

File handlers are specified using :ref:`shorthand syntax <shorthand_syntax>` in the :ref:`dataset <datasets>` configuration.
The following configuration utilizes the :class:`TorchFileHandler` to load and save PyTorch tensors with the file extension :file:`.pt`:

.. code-block:: yaml
   :caption: conf/dataset/ExampleDataset.yaml
   :linenos:
   
   ...
   file_type: pt
   file_handler: torch_file_handler.TorchFileHandler
   ...

.. _tut_target_transforms:

Custom Target Transforms
~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom :ref:`target transform <dataset_target_transforms>`, inherit from :class:`~autrainer.datasets.utils.AbstractTargetTransform`
and implement the :meth:`~autrainer.datasets.utils.AbstractTargetTransform.encode`, :meth:`~autrainer.datasets.utils.AbstractTargetTransform.decode`,
:meth:`~autrainer.datasets.utils.AbstractTargetTransform.predict_batch`,
and :meth:`~autrainer.datasets.utils.AbstractTargetTransform.majority_vote` methods.

For example, the following target transform logarithmically encodes and decodes the targets for regression tasks:

.. literalinclude:: ../examples/tutorials/log_target_transform.py
   :language: python
   :caption: log_target_transform.py
   :linenos:

The :ref:`target transforms <dataset_target_transforms>` are specified in the :attr:`~autrainer.datasets.AbstractDataset.target_transform` property
of a :ref:`dataset <datasets>` implementation.

.. _tut_advanced:

Advanced Data Pipelines
~~~~~~~~~~~~~~~~~~~~~~~

To create data and model pipelines that go beyond the standard :class:`~autrainer.datasets.utils.DataItem` convention of using
only :attr:`DataItem.features` as input to the model, first create a new :class:`~autrainer.datasets.utils.DataItem` struct
decorated with :class:`dataclasses.dataclass`:

.. literalinclude:: ../examples/tutorials/multi_branch_data.py
   :language: python
   :caption: multi_branch_data.py
   :lines: 14-19

and override :class:`~autrainer.datasets.utils.AbstractDataBatch`:

.. literalinclude:: ../examples/tutorials/multi_branch_data.py
   :language: python
   :caption: multi_branch_data.py
   :lines: 22-41

Following that, inherit from :class:`~autrainer.datasets.utils.DatasetWrapper`
to create a :class:`torch.utils.data.Dataset`
that iterates over your data 
and returns your custom :class:`~autrainer.datasets.utils.AbstractDataBatch`
(here we simply replicate :attr:`features`
as our auxiliary features):

.. literalinclude:: ../examples/tutorials/multi_branch_data.py
   :language: python
   :caption: multi_branch_data.py
   :lines: 44-52

Subsequently, inherit from :class:`~autrainer.datasets.AbstractDataset`
to create a dataset
that instantiates your :class:`~autrainer.datasets.utils.DatasetWrapper`:


.. literalinclude:: ../examples/tutorials/multi_branch_data.py
   :language: python
   :caption: multi_branch_data.py
   :lines: 55-73

Next, create a :attr:`ToyMultiBranch-C.yaml` configuration file for the dataset in the :attr:`conf/dataset/` directory:

.. literalinclude:: ../examples/tutorials/MultiBranchData-C.yaml
   :language: yaml
   :caption: conf/data/ToyMultiBranch-C.yaml
   :linenos:

Finally, you must inherit from :class:`~autrainer.models.AbstractModel` 
and create a model **with a matching signature**,
in its forward pass to access the :attr:`meta` parameter:

.. literalinclude:: ../examples/tutorials/multi_branch_model.py
   :language: python
   :caption: multi_branch_model.py
   :lines: 6-26


Next, create a 
:attr:`ToyMultiBranchModel.yaml` 
configuration file 
for the model in the :attr:`conf/model/` directory:


.. literalinclude:: ../examples/tutorials/MultiBranchModel.yaml
   :language: yaml
   :caption: conf/data/ToyMultiBranchModel.yaml
   :linenos:


.. _tut_optimizers:

Custom Optimizers
-----------------

To create a custom :ref:`optimizer <optimizers>`, inherit from
`torch.optim.Optimizer <https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer>`_ and implement the :meth:`step` method.

For example, the following optimizer implements the `SGD optimizer <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_
with an additional randomly scaled learning rate using a :ref:`custom step function <optimizer_custom_step>`:

.. literalinclude:: ../examples/tutorials/random_scaled_sgd.py
   :language: python
   :caption: random_scaled_sgd.py
   :linenos:


The following configuration creates a new :file:`RandomScaledSGD.yaml` optimizer in the :file:`conf/optimizer/` directory and uses the global
seed of the :ref:`main configuration <main_configuration>` as the :attr:`generator_seed` attribute:

.. literalinclude:: ../examples/tutorials/RandomScaledSGD.yaml
   :language: yaml
   :caption: conf/optimizer/RandomScaledSGD.yaml
   :linenos:

.. note::
   The :attr:`params` and :attr:`lr` attributes are automatically passed to the optimizer during initialization
   and determined at runtime.


.. _tut_schedulers:

Custom Schedulers
-----------------

To create a custom :ref:`scheduler <schedulers>`, inherit from
`torch.optim.lr_scheduler.LRScheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
and implement the :meth:`get_lr` method.

For example, the following scheduler implements a simple linear warm-up scheduler:

.. literalinclude:: ../examples/tutorials/linear_warm_up_lr.py
   :language: python
   :caption: linear_warm_up_lr.py
   :linenos:

The following configuration creates a new :file:`LinearWarmUpLR.yaml` scheduler with a linear warm-up period of 10
:ref:`training iterations <training>` in the :file:`conf/scheduler/` directory:

.. literalinclude:: ../examples/tutorials/LinearWarmUpLR.yaml
   :language: yaml
   :caption: conf/scheduler/LinearWarmUpLR.yaml
   :linenos:

.. note::
   The :attr:`optimizer` attribute is automatically passed to the scheduler during initialization and determined at runtime.


.. _tut_transforms:

Custom Transforms
-----------------

To create a custom :ref:`transform <transforms>`, inherit from :class:`~autrainer.transforms.AbstractTransform`
and implement the :meth:`~autrainer.transforms.AbstractTransform.__call__` method.

For example, the following transform denoises a spectrogram by applying a
`median filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html>`_:

.. literalinclude:: ../examples/tutorials/spect_median_filter.py
   :language: python
   :caption: spect_median_filter.py
   :linenos:

This transform can be used both as a :ref:`preprocessing transform <tut_preprocessing_transforms>` and as an
:ref:`online transform <tut_online_transforms>`.

.. _tut_preprocessing_transforms:

Custom Preprocessing Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom :ref:`preprocessing transform <preprocessing_transforms>`, create a new file in the :file:`conf/preprocessing/` directory.

For example, the following preprocessing transform extracts log-Mel spectrograms from audio data at a sampling rate of 32 kHz and
applies the :ref:`custom denoising transform <tut_transforms>` to the data:

.. literalinclude:: ../examples/tutorials/denoised_log_mel_32k.yaml
   :language: yaml
   :caption: conf/scheduler/denoised_log_mel_32k.yaml
   :linenos:

Any audio :ref:`dataset <datasets>` can utilize this preprocessing transform by specifying the :attr:`features_subdir`
attribute in the dataset configuration and adjusting the :attr:`file_type`, :attr:`file_handler`, and :attr:`transform` attributes:

.. code-block:: yaml
   :caption: conf/dataset/ExampleDataset.yaml
   :linenos:

   ...
   features_subdir: denoised_log_mel_32k
   file_type: npy
   file_handler: autrainer.datasets.utils.NumpyFileHandler
   ...
   transform:
     type: grayscale

.. note::
   
   The :meth:`~autrainer.datasets.utils.AbstractFileHandler.save` method of the :attr:`file_handler` specified in the dataset configuration
   is used to save the processed data to the :attr:`features_subdir` directory.
   The :meth:`~autrainer.datasets.utils.AbstractFileHandler.load` method of the :attr:`file_handler` is used to load the processed data
   during training and inference.


.. _tut_online_transforms:

Custom Online Transforms
~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom :ref:`online transform <online_transforms>`, no configuration file is necessary as the transform is applied at runtime
and specified in the :attr:`transform` attribute of the model and dataset configurations using :ref:`shorthand syntax <shorthand_syntax>`.

For example, the following configuration applies the :ref:`custom denoising transform <tut_transforms>` to the data at runtime:

.. code-block:: yaml
   :caption: conf/dataset/ExampleDataset.yaml
   :linenos:

   ...
   transform:
     type: grayscale
     base:
       - spect_median_filter.SpectMedianFilter:
           size: 5


In line with the :ref:`custom preprocessing transform <tut_preprocessing_transforms>` example,
the :ref:`custom denoising transform <tut_transforms>` is applied to the train, dev, and test datasets.

It may be desirable to only apply a transform to a specific subset of the data.
The following configuration applies the :ref:`custom denoising transform <tut_transforms>` only to the :attr:`train` subset of the data:

.. code-block:: yaml
   :caption: conf/dataset/ExampleDataset.yaml
   :linenos:

   ...
   transform:
     type: grayscale
     train:
       - spect_median_filter.SpectMedianFilter:
           size: 5


.. _tut_augmentations:

Custom Augmentations
--------------------

To create a custom :ref:`augmentation <augmentations>`, inherit from :class:`~autrainer.augmentations.AbstractAugmentation`
and implement the :meth:`~autrainer.augmentations.AbstractAugmentation.apply` method.

For example, the following augmentation scales the amplitude of a spectrogram by a random factor in a given range:

.. literalinclude:: ../examples/tutorials/amplitude_scale_augmentation.py
   :language: python
   :caption: amplitude_scale_augmentation.py
   :linenos:

The following configuration creates a new :file:`AmplitudeScale.yaml` augmentation in the :file:`conf/augmentation/` directory, 
scaling the amplitude of the spectrogram by a random factor between 0.8 and 1.2 with a probability :attr:`p` of 0.5:

.. literalinclude:: ../examples/tutorials/AmplitudeScale.yaml
   :language: yaml
   :caption: conf/augmentation/AmplitudeScale.yaml
   :linenos:

As no augmentation in the :attr:`pipeline` specifies a :attr:`generator_seed` attribute,
the global :attr:`generator_seed` attribute is broadcasted to all augmentations to ensure reproducibility.


.. _tut_augmentation_graphs:

Custom Augmentation Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, the following configuration creates a new :file:`AmplitudeScaleOrTimeFreqMask.yaml` augmentation in the
:file:`conf/augmentation/` directory, either applying the :ref:`custom amplitude scale augmentation <tut_augmentations>` or a sequence of the
:class:`~autrainer.augmentations.TimeMask` and :class:`~autrainer.augmentations.FrequencyMask` augmentations:

.. literalinclude:: ../examples/tutorials/AmplitudeScaleOrTimeFreqMask.yaml
   :language: yaml
   :caption: conf/augmentation/AmplitudeScaleOrTimeFreqMask.yaml
   :linenos:

The :ref:`custom amplitude scale augmentation <tut_augmentations>` is selected with a probability of 0.2,
while the sequence of the :class:`~autrainer.augmentations.TimeMask` and :class:`~autrainer.augmentations.FrequencyMask`
augmentations is selected with a probability of 0.8.


.. _tut_augmentation_collate:

Custom Collate Augmentations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom :ref:`collate augmentation <augmentation_collate>`, inherit from :class:`~autrainer.augmentations.AbstractAugmentation`
and implement the optional :meth:`~autrainer.augmentations.AbstractAugmentation.get_collate_fn` method.

The collate function is used to apply the augmentation on the batch level.
In case the collate function modifies the shape of the input or labels, this may need to be accounted for if the augmentation is not applied.

For example, the following augmentation randomly applies :class:`~autrainer.augmentations.CutMix` or :class:`~autrainer.augmentations.MixUp` augmentations
on the batch level:

.. literalinclude:: ../examples/tutorials/cut_mix_up.py
   :language: python
   :caption: cut_mix_up.py
   :linenos:

.. _tut_loggers:

Custom Loggers
--------------

To create a custom :ref:`logger <loggers>`, inherit from :class:`~autrainer.loggers.AbstractLogger` and implement the
:meth:`~autrainer.loggers.AbstractLogger.log_params`,
:meth:`~autrainer.loggers.AbstractLogger.log_metrics`,
:meth:`~autrainer.loggers.AbstractLogger.log_timers`,
and :meth:`~autrainer.loggers.AbstractLogger.log_artifact` methods, as well as the optional
:meth:`~autrainer.loggers.AbstractLogger.setup`, and :meth:`~autrainer.loggers.AbstractLogger.end_run` methods.

All methods are automatically called a the appropriate time during training and inference.

For example, the following logger logs to `Weights & Biases <https://wandb.ai/>`_:

.. literalinclude:: ../examples/tutorials/wandb_logger.py
   :language: python
   :caption: wandb_logger.py
   :linenos:

Note that the :class:`WandBLogger` assumes that `wandb <https://wandb.ai/>`_ is installed, the API key is set,
and a project with the same name as the :attr:`experiment_id` of the :ref:`main configuration <main_configuration>` exists.

To add the :class:`WandBLogger`, specify it in the :ref:`main configuration <main_configuration>` by adding a list of :attr:`loggers`:

.. code-block:: yaml
   :caption: conf/config.yaml
   :linenos:

   ...
   loggers:
     - wandb_logger.WandBLogger:
         output_dir: ${results_dir}/.wandb
   ...


.. _tut_callbacks:

Custom Callbacks
----------------

To create a custom :ref:`callback <callbacks>`, implement a class that specifies any of the callback functions defined in
:class:`~autrainer.training.CallbackSignature`.

For example, the following callback tracks learning rate changes at the beginning of each iteration:

.. literalinclude:: ../examples/tutorials/lr_tracker_callback.py
   :language: python
   :caption: lr_tracker_callback.py
   :linenos:

To add the :class:`LRTrackerCallback`, specify it in the :ref:`main configuration <main_configuration>` by adding a list of :attr:`callbacks`:

.. code-block:: yaml
   :caption: conf/config.yaml
   :linenos:

   ...
   callbacks:
     - lr_tracker_callback.LRTrackerCallback
   ...


.. _tut_plotting:

Custom Plotting
---------------

To create a custom :ref:`plotting <core_plotting>` configuration, create a new file in the :file:`conf/plotting/` directory.

For example, the following configuration uses the `LaTeX <https://matplotlib.org/stable/users/explain/text/usetex.html>`_ backend,
the Palatino font with a font size of 9, replaces :attr:`None` values in the run name with :attr:`~~` for better readability,
and adds labels as well as titles to the plot.

.. literalinclude:: ../examples/tutorials/LaTeX.yaml
   :language: yaml
   :caption: conf/plotting/LaTeX.yaml
   :linenos:

To add the :file:`LaTeX.yaml` plotting configuration, specify it in the :ref:`main configuration <main_configuration>`
by overriding the :attr:`plotting` attribute:

.. code-block:: yaml
   :caption: conf/config.yaml
   :linenos:

   defaults:
     - ...
     - override plotting: LaTeX
   ...


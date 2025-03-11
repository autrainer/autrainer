.. _transforms:

Transforms
==========

Transforms serve the purpose of preprocessing the input data before it is fed to the model and are specified as pipelines with :ref:`shorthand_syntax`.
This can be done both :ref:`offline <preprocessing_transforms>` (preprocessing) and :ref:`online <online_transforms>`.

.. tip::

   To create custom transforms, refer to the :ref:`custom transforms tutorial <tut_transforms>`.

Multiple transforms are combined into a pipeline, which is handled by the :class:`TransformManager`.
Pipelines are automatically assembled and sorted based on the :attr:`order` attribute of each transform, which is handled by the :class:`SmartCompose`.

Transforms with a lower order are applied first, followed by transforms with a higher order.
If two transforms share the same :attr:`order`, they are applied in the order they are specified in the configuration.

Both types of transforms can utilize any of the :ref:`available transforms <available_transforms>` provided by `autrainer`,
or custom transforms inheriting from the :class:`AbstractTransform` class.

While the choice to use offline or online transforms depends on the use case and is a tradeoff between storage and computational costs,
both can be used in conjunction for maximum flexibility.


.. _preprocessing_transforms:

Preprocessing Transforms
------------------------

Preprocessing transforms are specified in the dataset configuration aligning the :attr:`features_subdir` with the name of the preprocessing file.
This way, the processed data is stored in a subdirectory of the dataset directory with the same name as the preprocessing file,
or in a different path altogether by specifying the :attr:`features_path` attribute in the configuration file.
To save the processed data, the :attr:`file_handler` specified in the dataset configuration is used.

.. tip::

   To create custom preprocessing transforms, refer to the :ref:`custom preprocessing transforms tutorial <tut_preprocessing_transforms>`.

Preprocessing transforms consist of the following attributes:

* :attr:`file_handler`: The file handler to use for loading the data specified with :ref:`shorthand_syntax`.
  To discover all available file handlers, refer to the :ref:`file handlers <dataset_file_handlers>` section.
* :attr:`pipeline`: The sequence of transformations to apply to the data.
  The pipeline is specified using :ref:`shorthand_syntax` and can include any of the :ref:`available transforms <available_transforms>`.

.. note::
   To avoid race conditions during parallel training, the preprocessing is applied to the entire dataset before training.
   
   To automatically preprocess all datasets specified in the :ref:`main configuration <main_configuration>` (:file:`conf/config.yaml`) file,
   the following :ref:`preprocessing <cli_preprocessing>` CLI command can be used:

   .. code-block:: autrainer

      autrainer preprocess

   Alternatively, use the following :ref:`preprocessing <cli_wrapper_preprocessing>` CLI wrapper function:

   .. code-block:: python

      autrainer.cli.preprocess()

.. warning::
   Following `v0.6.0`, `autrainer` iterates over all files in all datasets
   and extracts the respective features without replacement.
   It is possible that features have been extracted for a subset of the dataset.
   This may happen when different dataset subsets are supported.
   Therefore, it is recommended that `autrainer preprocess` is always called before training.
   Features have to be manually deleted to overwrite.

`autrainer` offers default configurations for log-Mel spectrogram extraction and openSMILE feature extraction.

.. dropdown:: Default Configurations
   
   **Log-Mel Spectrograms**

   Log-Mel spectrograms are a common representation of audio data.

   `autrainer` offers default configurations for log-Mel spectrogram extraction for different sampling rates.
   The generation process is adapted from:
   `PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition <https://arxiv.org/abs/1912.10211>`_.


   .. configurations::
      :subdir: preprocessing
      :configs: log_mel

   **openSMILE Features**

   `openSMILE <https://audeering.github.io/opensmile-python/>`_ is a widely used feature extraction tool for audio data.

   `autrainer` provides default configurations for extracting openSMILE features.
   openSMILE is an optional dependency and may need to be installed separately.
   For installation, refer to the :ref:`installation guide <installation>`.

   .. configurations::
      :subdir: preprocessing
      :configs: eGeMAPS ComParE


.. _online_transforms:

Online Transforms
-----------------

Online transforms can be specified with the :attr:`transform` attribute in both the model and dataset,
applied to the input data before it is passed to the model.
Each pipeline is specified as a list of transforms using :ref:`shorthand_syntax`.

Model and dataset configurations specify the transforms to be applied to the input data.
If :ref:`augmentations <augmentations>` are used, they are merged with the online transforms to create a single pipeline.

.. tip::

   To create custom online transforms, refer to the :ref:`custom online transforms tutorial <tut_online_transforms>`.


Types
~~~~~

Each :attr:`transform` configuration includes a :attr:`type` attribute, specifying the type of data the model expects or the dataset provides:

* :attr:`image`: RGB images
* :attr:`grayscale`: Grayscale images (e.g., spectrograms or other single-channel images)
* :attr:`raw`: Raw audio waveforms
* :attr:`tabular`: Tabular data (e.g., openSMILE features)

.. note::
   The conversion between RGB and grayscale images is handled automatically by the pipeline to ensure compatibility between models and datasets.


Pipelines
~~~~~~~~~

Different transforms can be specified for :attr:`train`, :attr:`dev`, and :attr:`test` pipelines.
In addition, :attr:`base` provides a common set of transforms to include in all pipelines.

For example, the following configuration applies different :class:`~autrainer.transforms.RandomCrop`
transforms for the :attr:`train`, :attr:`dev`, and :attr:`test` pipelines (which may not necessarily be useful in practice):

.. code-block:: yaml
   :caption: conf/model/SpectrogramModel.yaml
   :linenos:

   ...
   transform:
     type: grayscale
     train:
       - autrainer.transforms.RandomCrop:
           size: 111
     dev:
       - autrainer.transforms.RandomCrop:
           size: 121
     test:
       - autrainer.transforms.RandomCrop:
           size: 131


Removing and Overriding
~~~~~~~~~~~~~~~~~~~~~~~

Models and datasets can remove any existing transform if it exists to ensure compatibility between datasets and models.
Removal is done by setting the transform to :attr:`null` in the configuration.
If a transform is set to :attr:`null` but is not present in the pipeline, it is ignored.

For example, a model taking in spectrograms could set the :class:`~autrainer.transforms.Normalize` transform to :attr:`null`
to remove it from the pipeline as it is not needed for spectrograms:

.. code-block:: yaml
   :caption: conf/model/SpectrogramModel.yaml
   :linenos:

   ...
   transform:
     type: grayscale
     base:
       - autrainer.transforms.Normalize: null


Model :ref:`online transforms <online_transforms>` outweigh dataset :ref:`online transforms <online_transforms>`.
If a model specifies a transform that is also specified in the dataset,
the model transform is used, overriding the dataset transform.

.. note::
   Removing and overriding transforms is useful for ensuring compatibility between models and datasets, however,
   both are bound to the same pipeline (e.g. :attr:`base`, :attr:`train`, :attr:`dev`, or :attr:`test`).

   If multiple transforms of the same type are specified in a pipeline, overriding or removing a transform affects all transforms
   of that type in the pipeline.


Tags
~~~~

In case multiple transforms of the same type are specified in the pipeline, they can be tagged with a unique identifier
using an :attr:`@` symbol followed by the tag name.

For example, the following configuration applies two :class:`~autrainer.transforms.Normalize` transforms to the pipeline,
each with a different tag:

.. code-block:: yaml
   :caption: conf/model/SpectrogramModel.yaml
   :linenos:

   ...
   transform:
     type: grayscale
     base:
       - autrainer.transforms.Normalize@first:
           mean: 0.5
           std: 0.5
       - autrainer.transforms.Normalize@second:
           mean: 123
           std: 456


Transforms with tags can be removed or overridden analogously to transforms without tags, by appending the tag name to the transform name.

.. note::

   If a transform with a tag is removed or overridden, only the transform with the specified tag (if it exists) is affected,
   allowing for the removal or overriding of specific transforms in the pipeline.


Transform Manager
-----------------

The :class:`TransformManager` is responsible for building the transformation pipeline based on the model and dataset configurations.
In addition, it handles the inclusion of augmentations.

.. autoclass:: autrainer.transforms.TransformManager
   :members:

Smart Compose
-------------

The :class:`SmartCompose` is a helper that allows for the composition and ordering of transformations.

.. autoclass:: autrainer.transforms.SmartCompose
   :members:
   :special-members: __add__, __call__


Abstract Transform
------------------

All transforms must inherit from :class:`AbstractTransform` and implement the :meth:`__call__` method.

.. autoclass:: autrainer.transforms.AbstractTransform
   :members:
   :special-members: __call__


.. _available_transforms:

Available Transforms
--------------------

`autrainer` provides a set of predefined transforms that can be used in any configuration.

.. automodule:: autrainer.transforms
   :members:
   :exclude-members: AbstractTransform, SmartCompose, TransformManager

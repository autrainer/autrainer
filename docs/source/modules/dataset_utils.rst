Dataset Utils
=============

Dataset utilities provide :ref:`file handlers <dataset_file_handlers>`, :ref:`target (or label) transforms <dataset_target_transforms>`,
and a :ref:`dataset wrapper <dataset_wrapper>` for :ref:`datasets <datasets>`.


.. _dataset_file_handlers:

File Handlers
-------------

File handlers are used to load and save files and are specified using :ref:`shorthand syntax <shorthand_syntax>`
in the :ref:`dataset <datasets>` and :ref:`preprocessing <preprocessing_transforms>` configurations.

.. tip::

   To create custom file handlers, refer to the :ref:`custom file handlers tutorial <tut_file_handlers>`.


.. autoclass:: autrainer.datasets.utils.AbstractFileHandler
   :members:

.. autoclass:: autrainer.datasets.utils.AudioFileHandler
   :members:

.. autoclass:: autrainer.datasets.utils.IdentityFileHandler
   :members:

.. autoclass:: autrainer.datasets.utils.ImageFileHandler
   :members:

.. autoclass:: autrainer.datasets.utils.NumpyFileHandler
   :members:


.. _dataset_target_transforms:

Target Transforms
------------------

Target transforms are specified using :ref:`shorthand syntax <shorthand_syntax>` in the dataset configuration 
and used to encode as well as decode the targets (or labels) of the dataset.
Additionally, target transforms provide functiuons for batch prediction and majority voting.

.. tip::

   To create custom target transforms, refer to the :ref:`custom target transforms tutorial <tut_target_transforms>`.


.. autoclass:: autrainer.datasets.utils.AbstractTargetTransform
   :members:

.. autoclass:: autrainer.datasets.utils.LabelEncoder
   :members:

.. autoclass:: autrainer.datasets.utils.MinMaxScaler
   :members:

.. autoclass:: autrainer.datasets.utils.MultiLabelEncoder
   :members:


.. _dataset_wrapper:

Dataset Wrapper
------------------

The :class:`DatasetWrapper` provides a wrapper around a :class:`torch.utils.data.Dataset`, utilizing the :ref:`file handlers <dataset_file_handlers>`
and :ref:`target transforms <dataset_target_transforms>` to load and transform the data, returning the data, target (or label), and index of each sample.

.. autoclass:: autrainer.datasets.utils.DatasetWrapper
   :members:
   :special-members: __getitem__

ZIP Downloader
------------------

To automatically download and extract zip files, the :class:`~autrainer.datasets.utils.ZipDownloadManager` can be used.

.. autoclass:: autrainer.datasets.utils.ZipDownloadManager
   :members:
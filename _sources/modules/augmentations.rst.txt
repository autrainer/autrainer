.. _augmentations:

Augmentations
=============

Augmentations are optional and by default not used.
This is indicated by the absence of the :attr:`augmentation`
attribute in the sweeper configuration (implicitly set to a :file:`None` configuration file).
To use an augmentation, specify it in the configuration file (:file:`conf/config.yaml`) for the sweeper.

.. tip::

   To create custom augmentations, refer to the :ref:`custom augmentations tutorial <tut_augmentations>`.

Augmentations are specified analogously to :ref:`transforms <transforms>` using :ref:`shorthand syntax <shorthand_syntax>`
and have an :attr:`order` attribute to define the order of the augmentations.
The augmentations are combined with the transform pipeline and sorted based on the order of the augmentations as well as the transforms.

In addition to the order of the augmentation, a seeded probability :attr:`p` of applying the augmentation can be specified.
The optional :attr:`generator_seed` attribute is used to seed the random number generator for the augmentation.

.. dropdown:: Default Configurations
   
   **None**
   
   This configuration file is used to indicate that no augmentation is used and serves as a no-op placeholder.

   .. configurations::
      :subdir: augmentation
      :configs: None
      :exact:

Augmentation Pipelines
----------------------

The :class:`~autrainer.augmentations.AugmentationManager` is responsible for building the augmentation pipeline.


.. autoclass:: autrainer.augmentations.AugmentationManager
    :members:

The :class:`~autrainer.augmentations.AugmentationPipeline` class is used to define the configuration and instantiate the augmentation pipeline.

.. autoclass:: autrainer.augmentations.AugmentationPipeline
    :members:


Abstract Augmentation
---------------------

.. autoclass:: autrainer.augmentations.AbstractAugmentation
    :special-members: __call__
    :members:


Augmentation Wrappers
---------------------

For easier access to common augmentation libraries, `autrainer` provides wrappers for
`torchaudio <https://pytorch.org/audio/stable/transforms.html>`_,
`torchvision <https://pytorch.org/vision/stable/transforms.html>`_,
`torch-audiomentations <https://github.com/asteroid-team/torch-audiomentations>`_,
and `albumentations <https://albumentations.ai/docs/>`_ augmentations.

The underlying augmentation is specified with the :attr:`name` attribute, representing the class name of the augmentation.
Any further attributes are passed as keyword arguments to the augmentation constructor.  

.. note::
   For each augmentation, the probability :attr:`p` of applying the augmentation is always available, if the underlying augmentation supports it.
   If not specified, the default value is 1.0, overriding any existing default value of the library.

Both `torch-audiomentations` and `albumentations` augmentations are optional and can be installed using the following commands.

.. code-block:: pip

   pip install autrainer[albumentations]
   pip install autrainer[torch-audiomentations]

.. autoclass:: autrainer.augmentations.TorchaudioAugmentation

   .. dropdown:: Default Configurations

      No default configurations are provided for `torchaudio` augmentations.
      To discover the available `torchaudio` augmentations,
      refer to the `torchaudio documentation <https://pytorch.org/audio/stable/transforms.html>`_.

.. autoclass:: autrainer.augmentations.TorchvisionAugmentation

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: AugMix GaussianBlur RandAugment RandGrayscale RandomRotation
         :headline:

.. autoclass:: autrainer.augmentations.AudiomentationsAugmentation

   .. dropdown:: Default Configurations

      No default configurations are provided for `audiomentations` augmentations.
      To discover the available `audiomentations` augmentations,
      refer to the `audiomentations documentation <https://iver56.github.io/audiomentations/>`_.

.. autoclass:: autrainer.augmentations.TorchAudiomentationsAugmentation

   .. dropdown:: Default Configurations

      No default configurations are provided for `torch-audiomentations` augmentations.
      To discover the available `torch-audiomentations` augmentations,
      refer to the `torch-audiomentations documentation <https://github.com/asteroid-team/torch-audiomentations>`_.

.. autoclass:: autrainer.augmentations.AlbumentationsAugmentation

   .. dropdown:: Default Configurations

      No default configurations are provided for `albumentations` augmentations.
      To discover the available `albumentations` augmentations,
      refer to the `albumentations documentation <https://albumentations.ai/docs/>`_.


Augmentation Graphs
-------------------

To create more complex augmentation pipelines which may resemble a graph structure, :class:`~autrainer.augmentations.Sequential`
and :class:`~autrainer.augmentations.Choice` can be used.

.. tip::

   To create custom augmentation graphs, refer to the :ref:`custom augmentation graphs tutorial <tut_augmentation_graphs>`.


.. autoclass:: autrainer.augmentations.Sequential
    :members:

.. autoclass:: autrainer.augmentations.Choice
    :members:

.. note::
   The order of :class:`~autrainer.augmentations.Sequential` and :class:`~autrainer.augmentations.Choice`
   can be defined in the configuration file by the :attr:`order` attribute.
   However, order attributes of the augmentations within the :class:`~autrainer.augmentations.Sequential`
   and :class:`~autrainer.augmentations.Choice` are ignored.
   As the augmentations are applied in a scoped manner, their order is determined by the order of the augmentations in the configuration file.


Spectrogram Augmentations
-------------------------

.. autoclass:: autrainer.augmentations.GaussianNoise

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: GaussianNoise

.. autoclass:: autrainer.augmentations.TimeMask

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: TimeMask

.. autoclass:: autrainer.augmentations.FrequencyMask

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: FrequencyMask

.. autoclass:: autrainer.augmentations.TimeShift

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: TimeShift

.. autoclass:: autrainer.augmentations.TimeWarp

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: TimeWarp

.. autoclass:: autrainer.augmentations.SpecAugment

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: SpecAugment


.. _augmentation_collate:

Collate Functions
-----------------

For augmentations that require a collate function, an optional :attr:`get_collate_fn` method can be implemented.
This method is used to retrieve the collate function from the augmentation if it is present.

.. tip::

   To create custom augmentations with collate functions, refer to the :ref:`custom augmentations tutorial <tut_augmentation_collate>`.

The signature of the :attr:`get_collate_fn` method should be as follows and return a collate function taking in a list of
:class:`~autrainer.core.structs.AbstractDataItem` objects and returning a single :class:`~autrainer.core.structs.AbstractDataBatch` object.

.. literalinclude:: ../examples/augmentation_get_collate_fn.py
   :language: python
   :caption: example_augmentation.ExampleCollateAugmentation
   :linenos:
   :lines: 11-

.. note::
   Only one collate function can be used in each transform pipeline.
   If multiple collate functions are defined, the last one in the pipeline (defined by the order of the :ref:`transforms <transforms>`) is used.

Both :class:`~autrainer.augmentations.CutMix` and :class:`~autrainer.augmentations.MixUp`
augmentations require a collate function and operate on the batch level.
This means, that the collate function is applied to the batch of samples, rather than individual samples,
and the probability of applying the augmentation acts on the batch level as well.

.. autoclass:: autrainer.augmentations.CutMix

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: CutMix

.. autoclass:: autrainer.augmentations.MixUp

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: MixUp


Miscellaneous Augmentations
---------------------------

.. autoclass:: autrainer.augmentations.SampleGaussianWhiteNoise

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: augmentation
         :configs: SampleGaussianWhiteNoise


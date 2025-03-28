.. _models:

Models
======

`autrainer` provides a number of different audio-specific models as well as wrappers for
`torchvision <https://pytorch.org/vision/stable/models.html>`_ and `timm <https://timm.fast.ai/>`_ models.

.. tip::

   To create custom models, refer to the :ref:`custom models tutorial <tut_models>`.

Default configurations that end in :file:`-T` indicate that the model uses transfer learning with pretrained weights.
To avoid race conditions when using :ref:`hydra_launcher_plugins` that may run multiple training jobs in parallel,
:ref:`autrainer fetch <cli_autrainer_fetch>` or :meth:`~autrainer.cli.fetch` are used to download the pretrained weights before training.

.. note::
   
   Most models are pretrained on the `ImageNet <http://www.image-net.org/>`_ or `AudioSet <https://research.google.com/audioset/>`_ datasets.
   To ensure compatibility with any number of output dimensions, the last linear layer of the model is replaced with a new linear layer with
   the correct number of output dimensions and will therefore **not** be pretrained.

   The weights for all pretrained models that are provided by `autrainer` can be automatically downloaded using the
   :ref:`autrainer fetch <cli_autrainer_fetch>` CLI command or the :meth:`~autrainer.cli.fetch` CLI wrapper function.

To optionally use model, optimizer, or scheduler checkpoints, the following attributes can be set in any model configuration file:

* :attr:`model_checkpoint`: The path to the model checkpoint file. Defaults to None.
* :attr:`optimizer_checkpoint`: The path to the optimizer checkpoint file. Defaults to None.
* :attr:`scheduler_checkpoint`: The path to the scheduler checkpoint file. Defaults to None.
* :attr:`skip_last_layer`: Whether to skip loading the state of the last linear or convolutional layer.
  When set to True, the state of the last layer (if present) is omitted from both the model and optimizer,
  allowing for training on a different target dataset with varying output dimensions. 
  Defaults to True.

.. note::

   Loading a checkpoint assumes that the model architecture is the same as the one used to create the checkpoint and
   that the last layer of the model is specified as the final :class:`~torch.nn.Linear` or :class:`~torch.nn.modules.conv._ConvNd` module.
   If the last layer is not the final layer in the module order, it may not be correctly identified for skipping.


Abstract Model
--------------

All models inherit from the :class:`AbstractModel` class and implement the :meth:`forward` and :meth:`embeddings` methods.
These models must adhere 
a specific signature 
for their :meth:`forward` method
as `autrainer` uses 
argument names
for its internal data flow.
For this reason,
:attr:`features`
must always be the name
of the first argument.
All subsequent arguments
can be named differently
but they must be present
as attributes
in the corresponding
:class:`~autrainer.datasets.utils.AbstractDataItem`.

.. autoclass:: autrainer.models.AbstractModel
   :members:

Model Wrappers
--------------

For convenience, we provide wrappers for torchvision and timm models.

.. autoclass:: autrainer.models.TorchvisionModel

   .. dropdown:: Default Configurations
      
      `autrainer` provides default configurations for all torchvision classification models.
      For more information on the available torchvision models as well as their parameters,
      refer to the `torchvision classification models documentation <https://pytorch.org/vision/stable/models.html#classification>`_.

      By default, models using transfer learning (indicated by a trailing :file:`-T` in the model name)
      use the default pretrained weights provided by torchvision.

      .. configurations::
         :subdir: model
         :configs:
          AlexNet
          ConvNeXt
          Densenet
          EfficientNet
          GoogLeNet
          InceptionV3
          MaxViT
          MnasNet
          MobileNet
          RegNet
          ResNet
          ResNeXt
          ShuffleNet
          SqueezeNet
          Swin
          VGG
          MaxViT
          Wide-ResNet
         :headline:
         

.. autoclass:: autrainer.models.TimmModel

   .. dropdown:: Default Configurations
      
      `autrainer` does not provide default configurations for timm models.
      To discover the available timm models to create a custom configuration, refer to the `timm documentation <https://timm.fast.ai/>`_.

Audio Models
------------

`autrainer` provides a number of different audio-specific models.

.. autoclass:: autrainer.models.ASTModel

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: ASTModel

.. autoclass:: autrainer.models.AudioRNNModel

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: End2You

.. autoclass:: autrainer.models.Cnn10

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: Cnn10

.. autoclass:: autrainer.models.Cnn14

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: Cnn14

.. autoclass:: autrainer.models.FFNN

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: ToyFFNN

.. autoclass:: autrainer.models.LEAFNet

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: LEAFNet

.. autoclass:: autrainer.models.SeqFFNN

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: Seq-FFNN

.. autoclass:: autrainer.models.TDNNFFNN

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: TDNNFFNN

.. autoclass:: autrainer.models.W2V2FFNN

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: w2v2

.. autoclass:: autrainer.models.WhisperFFNN

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: model
         :configs: Whisper-FFNN

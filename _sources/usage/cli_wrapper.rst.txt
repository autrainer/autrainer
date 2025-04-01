.. _cli_wrapper:

CLI Wrapper
=============

`autrainer` provides an :attr:`autrainer.cli` CLI wrapper to programmatically manage the entire training process,
including :ref:`configuration management <cli_wrapper_configuration_management>`,
:ref:`data preprocessing <cli_wrapper_preprocessing>`,
:ref:`model training <cli_wrapper_training>`,
:ref:`inference <cli_wrapper_inference>`,
and :ref:`postprocessing <cli_wrapper_postprocessing>`.

Wrapper functions are useful for integrating `autrainer` into custom scripts, jupyter notebooks, google colab notebooks, and other applications.

In addition to the CLI wrapper functions, `autrainer` provides a :ref:`CLI <cli_reference>` to manage configurations, data, training, inference,
and postprocessing from the command line with the same functionality as the CLI wrapper.


.. _cli_wrapper_configuration_management:

Configuration Management
------------------------

To manage configurations, :meth:`~autrainer.cli.create`, :meth:`~autrainer.cli.list`,
and :meth:`~autrainer.cli.show` allow for the creation of the project structure and the discovery
as well as saving of default configurations provided by `autrainer`.

.. tip::
   
   Default configurations can be discovered both through the :ref:`CLI <cli_reference>`,
   the :ref:`CLI wrapper <cli_wrapper>`, and the respective module documentation.


.. _cli_wrapper_autrainer_create:

.. autofunction:: autrainer.cli.create

.. _cli_wrapper_autrainer_list:

.. autofunction:: autrainer.cli.list

.. _cli_wrapper_autrainer_show:

.. autofunction:: autrainer.cli.show


.. _cli_wrapper_preprocessing:

Preprocessing
-------------

To avoid race conditions when using :ref:`hydra_launcher_plugins` that may run multiple training jobs in parallel,
:meth:`~autrainer.cli.fetch` and :meth:`~autrainer.cli.preprocess` allow for
downloading and :ref:`preprocessing <preprocessing_transforms>` of :ref:`datasets` (and pretrained model states) before training.

Both commands are based on the :ref:`main configuration <main_configuration>` file (e.g., :file:`conf/config.yaml`),
such that the specified models and datasets are fetched and preprocessed accordingly.
If a model or dataset is already fetched or preprocessed, it will be skipped.

.. _cli_wrapper_autrainer_fetch:

.. autofunction:: autrainer.cli.fetch

.. _cli_wrapper_autrainer_preprocess:

.. autofunction:: autrainer.cli.preprocess


.. _cli_wrapper_training:

Training
--------

Training is managed by :meth:`~autrainer.cli.train`, which starts the training process
based on the :ref:`main configuration <main_configuration>` file (e.g., :file:`conf/config.yaml`).


.. _cli_wrapper_autrainer_train:

.. autofunction:: autrainer.cli.train


.. _cli_wrapper_inference:

Inference
---------

:meth:`~autrainer.cli.inference` allows for the (sliding window) inference of audio data using a trained model.

Both local paths and `Hugging Face Hub <https://huggingface.co/>`_ links are supported for the model.
Hugging Face Hub links are automatically downloaded and cached in the torch cache directory.

The following syntax is supported for Hugging Face Hub links: :code:`hf:repo_id[@revision][:subdir]#local_dir`.
This syntax consists of the following components:

* :code:`hf`: The Hugging Face Hub prefix indicating that the model is fetched from the Hugging Face Hub.
* :code:`repo_id`: The repository ID of the model consisting of the user name and the model card name separated by a slash
  (e.g., :code:`autrainer/example`).
* :code:`revision` (`optional`): The revision as a commit hash, branch name, or tag name (e.g., :code:`main`).
  If not specified, the latest revision is used.
* :code:`subdir` (`optional`): The subdirectory of the repository containing the model directory (e.g., :code:`AudioModel`).
  If not specified, the model directory is automatically inferred.
  If multiple models are present in the :code:`repo_id`, :code:`subdir` must be specified, as the correct model cannot be automatically inferred.
* :code:`local_dir` (`optional`): The local directory to which the model is downloaded to (e.g., :code:`.hf_local`).
  If not specified, the model is placed in the
  `torch hub cache directory <https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved>`_.

For example, to download the model from the repository :code:`autrainer/example`
at the revision :code:`main` from the subdirectory :code:`AudioModel` and save it to the local directory :code:`.hf_local`,
the following :meth:`~autrainer.cli.inference` CLI wrapper function can be used:

.. code-block:: python
  
   import autrainer.cli
   
   autrainer.cli.inference(
       model="hf:autrainer/example@main:AudioModel#.hf_local",
       input="input",
       output="output",
       device="cuda:0",
   )

.. tip::
   
      To access private repositories, the environment variable :code:`HF_HOME` should point to the
      `Hugging Face User Access Token <https://huggingface.co/docs/hub/security-tokens>`_.

      To use a custom endpoint (e.g., for a `self-hosted hub <https://huggingface.co/enterprise>`_),
      the environment variable :code:`HF_ENDPOINT` should point to the desired endpoint URL.

To use a local model path, the following :meth:`~autrainer.cli.inference` CLI wrapper function can be used:


.. code-block:: python
      
   import autrainer.cli
   
   autrainer.cli.inference(
       model="/path/to/AudioModel",
       input="input",
       output="output",
       device="cuda:0",
   )


.. _cli_wrapper_autrainer_inference:

.. autofunction:: autrainer.cli.inference


.. _cli_wrapper_postprocessing:

Postprocessing
--------------

Postprocessing allows for the summarization, visualization, and aggregation of the training results using :meth:`~autrainer.cli.postprocess`.
Several cleanup utilities are provided by :meth:`~autrainer.cli.rm_failed` and :meth:`~autrainer.cli.rm_states`.
Manual grouping of the training results can be done using :meth:`~autrainer.cli.group`.


.. _cli_wrapper_autrainer_postprocess:

.. autofunction:: autrainer.cli.postprocess

.. _cli_wrapper_autrainer_rm_failed:

.. autofunction:: autrainer.cli.rm_failed

.. _cli_wrapper_autrainer_rm_states:

.. autofunction:: autrainer.cli.rm_states

.. _cli_wrapper_autrainer_group:

.. autofunction:: autrainer.cli.group

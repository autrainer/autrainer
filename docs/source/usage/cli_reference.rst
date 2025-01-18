.. _cli_reference:

CLI Reference
=============

`autrainer` provides a command line interface (CLI) to manage the entire training process,
including :ref:`configuration management <cli_configuration_management>`,
:ref:`data preprocessing <cli_preprocessing>`,
:ref:`model training <cli_training>`,
:ref:`inference <cli_inference>`,
and :ref:`postprocessing <cli_postprocessing>`.

In addition to the CLI, `autrainer` provides an :ref:`CLI wrapper <cli_wrapper>` to manage configurations, data, training, inference,
and postprocessing programmatically with the same functionality as the CLI.

.. _cli_autrainer:

autrainer
---------

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :nosubcommands:

   .. code-block:: aucli

      usage: autrainer [-h] [-v] <command> ...


.. _cli_configuration_management:

Configuration Management
------------------------

To manage configurations, :ref:`autrainer create <cli_autrainer_create>`, :ref:`autrainer list <cli_autrainer_list>`,
and :ref:`autrainer show <cli_autrainer_show>` allow for the creation of the project structure and the discovery
as well as saving of default configurations provided by `autrainer`.

.. tip::
   
   Default configurations can be discovered both through the :ref:`CLI <cli_reference>`,
   the :ref:`CLI wrapper <cli_wrapper>`, and the respective module documentation.


.. _cli_autrainer_create:

autrainer create
~~~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: create

   .. code-block:: aucli

      usage: autrainer create [-h] [-e] [-a] [-f] [directories ...]


.. _cli_autrainer_list:

autrainer list
~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: list

   .. code-block:: aucli

      usage: autrainer list [-h] [-l] [-g] [-p P] directory


.. _cli_autrainer_show:

autrainer show
~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: show

   .. code-block:: aucli

      usage: autrainer show [-h] [-s] [-f] directory config


.. _cli_preprocessing:

Preprocessing
-------------

To avoid race conditions when using :ref:`hydra_launcher_plugins` that may run multiple training jobs in parallel,
:ref:`autrainer fetch <cli_autrainer_fetch>` and :ref:`autrainer preprocess <cli_autrainer_preprocess>` allow for
downloading and :ref:`preprocessing <preprocessing_transforms>` of :ref:`datasets` (and pretrained model states) before training.

Both commands are based on the :ref:`main configuration <main_configuration>` file (e.g., :file:`conf/config.yaml`),
such that the specified models and datasets are fetched and preprocessed accordingly.
If a model or dataset is already fetched or preprocessed, it will be skipped.


.. _cli_autrainer_fetch:

autrainer fetch
~~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: fetch

   .. code-block:: aucli

      usage: autrainer fetch [-h] [-b]


.. _cli_autrainer_preprocess:

autrainer preprocess
~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: preprocess

   .. code-block:: aucli

      usage: autrainer preprocess [-h] [-b] [-n N] [-p P] [-s]


.. _cli_training:

Training
--------

Training is managed by :ref:`autrainer train <cli_autrainer_train>`, which starts the training process
based on the :ref:`main configuration <main_configuration>` file (e.g., :file:`conf/config.yaml`).

.. _cli_autrainer_train:

autrainer train
~~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: train

   .. code-block:: aucli

      usage: autrainer train [-h]


.. _cli_inference:

Inference
---------

:ref:`autrainer inference <cli_autrainer_inference>` allows for the (sliding window) inference of audio data using a trained model.

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
the following :ref:`autrainer inference <cli_autrainer_inference>` CLI command can be used:

.. code-block:: autrainer

   autrainer inference hf:autrainer/example@main:AudioModel#.hf_local input/ output/ -d cuda:0

.. tip::
   
      To access private repositories, the environment variable :code:`HF_HOME` should point to the
      `Hugging Face User Access Token <https://huggingface.co/docs/hub/security-tokens>`_.

      To use a custom endpoint (e.g., for a `self-hosted hub <https://huggingface.co/enterprise>`_),
      the environment variable :code:`HF_ENDPOINT` should point to the desired endpoint URL.


To use a local model path, the following :ref:`autrainer inference <cli_autrainer_inference>` CLI command can be used:

.. code-block:: autrainer

   autrainer inference /path/to/AudioModel input/ output/ -d cuda:0


.. _cli_autrainer_inference:

autrainer inference
~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: inference

   .. code-block:: aucli

      usage: autrainer inference [-h] [-c C] [-d D] [-e E] [-r] [-emb] [-p P] [-w W] [-s S] [-m M] [-sr SR] model input output


.. _cli_postprocessing:

Postprocessing
--------------

Postprocessing allows for the summarization, visualization, and aggregation of the training results using :ref:`autrainer postprocess <cli_autrainer_postprocess>`.
Several cleanup utilities are provided by :ref:`autrainer rm-failed <cli_autrainer_rm_failed>` and :ref:`autrainer rm-states <cli_autrainer_rm_states>`.
Manual grouping of the training results can be done using :ref:`autrainer group <cli_autrainer_group>`.


.. _cli_autrainer_postprocess:

autrainer postprocess
~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: postprocess

   .. code-block:: aucli

      usage: autrainer postprocess [-h] [-m N] [-a A [A ...]] results_dir experiment_id


.. _cli_autrainer_rm_failed:

autrainer rm-failed
~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: rm-failed

   .. code-block:: aucli

      usage: autrainer rm-failed [-h] [-f] results_dir experiment_id


.. _cli_autrainer_rm_states:

autrainer rm-states
~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: rm-states

   .. code-block:: aucli

      usage: autrainer rm-states [-h] [-b] [-r R [R ...]] [-i I [I ...]] results_dir experiment_id


.. _cli_autrainer_group:

autrainer group
~~~~~~~~~~~~~~~

.. argparse::
   :module: autrainer.core.scripts.cli
   :func: get_parser
   :prog: autrainer
   :nodefault:
   :noepilog:
   :path: group

   .. code-block:: aucli

      usage: autrainer group [-h]

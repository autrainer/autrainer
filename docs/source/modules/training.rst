.. _training:

Training
========

`autrainer` supports both :attr:`epoch`- and :attr:`step`-based training.

.. tip::

   To create custom training configurations, refer to the :ref:`custom training configurations quickstart <quick_training_configurations>`.

Configuring Training
--------------------

Training is configured in the :ref:`main configuration <main_configuration>` file and comprises the following attributes:

* :attr:`iterations`: The number of iterations to train the model for.
* :attr:`training_type`: The type of training, either :attr:`epoch` or :attr:`step`. By default, it is set to :attr:`epoch`.
* :attr:`eval_frequency`: The frequency in terms of iterations to evaluate the model on the development set. By default, it is set to 1.
* :attr:`save_frequency`: The frequency in terms of iterations to states of the model, optimizer, and scheduler. By default, it is set to 1.
* :attr:`inference_batch_size`: The batch size to use during inference. By default, it is set to the training batch size.

The following optional attributes can be set to configure the training process:

* :attr:`progress_bar`: Whether to display a progress bar during training and evaluation. By default, it is set to True.
* :attr:`continue_training`: Whether to continue from an already finished run with the same configuration and fewer iterations.
  By default, it is set to True.
* :attr:`remove_continued_runs`: Whether to remove the runs that have been continued. By default, it is set to True.
* :attr:`save_train_outputs`: Whether to save indices, targets, losses, outputs, and predictions (results) on the training set.
  By default, it is set to True.
* :attr:`save_dev_outputs`: Whether to save indices, targets, losses, outputs, and predictions (results) on the development set.
  By default, it is set to True.
* :attr:`save_test_outputs`: Whether to save indices, targets, losses, outputs, and predictions (results) on the test set.
  By default, it is set to True.

For brevity, all training attributes with default values are outsourced to the :ref:`_autrainer_.yaml defaults <autrainer_defaults>`
file and imported in the :ref:`main configuration <main_configuration>` file.

.. note::
   Throughout the documentation, the term `iteration` (as well as :attr:`iterations`, :attr:`eval_frequency`, and :attr:`save_frequency`)
   refers to a full pass over the training set for epoch-based training, and a single optimization step over a batch of the training set
   for step-based training.


Trainer
-------

:class:`autrainer.training.Trainer` manages the training process.
It instantiates the model, dataset, criterion, optimizer, scheduler, and callbacks, and trains the model on the dataset.
It also logs the training process and saves the model, optimizer, and scheduler states at the end of each epoch.

The :attr:`cfg` of the trainer is the composed main configuration file (e.g. :file:`conf/config.yaml`) for each training configuration in the sweep.

.. autoclass:: autrainer.training.ModularTaskTrainer
   :members:

Callbacks
---------

Any model, dataset, criterion, ... of the trainer can specify callbacks.
Each callback is automatically called at the appropriate time during training.
Callbacks are functions of the same signature as any of the following callbacks:

.. autoclass:: autrainer.training.CallbackSignature
   :members:
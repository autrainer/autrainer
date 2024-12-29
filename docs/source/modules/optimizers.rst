.. _optimizers:

Optimizers
==========

Any :ref:`torch optimizer <torch_optimizers>` or :ref:`custom optimizer <custom_optimizers>` can be used for training.

.. tip::

   To create custom optimizers, refer to the :ref:`custom optimizers tutorial <tut_optimizers>`.


.. _torch_optimizers:

Torch Optimizers
----------------

Torch optimizers (:class:`torch.optim`) can be specified as relative python import paths for the :attr:`_target_` attribute in the configuration file.
Any additional attributes (except :attr:`id`) are passed as keyword arguments to the optimizer constructor.

For example, :class:`torch.optim.Adam` can be used as follows:

.. configurations::
      :subdir: optimizer
      :configs: Adam
      :exact:

`autrainer` provides a number of default configurations for torch optimizers:

.. dropdown:: Default Configurations

   .. configurations::
      :subdir: optimizer
      :configs: SGD Adam AdamW Adadelta Adagrad Adamax NAdam RAdam SparseAdam RMSprop Rprop ASGD LBFGS
      :exact:

.. _custom_optimizers:

Custom Optimizers
-----------------

.. autoclass:: autrainer.optimizers.SAM

 .. dropdown:: Default Configurations

    .. configurations::
       :subdir: optimizer
       :configs: SAM


.. _optimizer_custom_step:

Custom Step Function
--------------------

Custom optimizers can optionally provide a :func:`custom_step` function that is called instead
of the standard training step and should be defined as follows:

.. literalinclude:: ../examples/optimizer_custom_step.py
   :language: python
   :caption: `custom_step function of an optimizer`
   :linenos:
   :lines: 6-

.. note::

   The :func:`custom_step` function should perform both the forward and backward pass as well as update the model parameters accordingly.


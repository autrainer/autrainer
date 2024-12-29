.. _criterions:

Criterions
==========

Criterions are specified in the :ref:`Dataset<datasets>` configuration with :ref:`shorthand_syntax` and are used to calculate the loss of the model.

.. tip::

   To create custom criterions, refer to the :ref:`custom criterions tutorial <tut_criterions>`.


.. note::

   The :attr:`reduction` attribute of each criterion is automatically set to :attr:`"none"` during training, validation, and testing.
   This allows the per-example loss to be reported directly, without the need for re-calculating the loss for logging purposes.

   This is handled automatically during the instantiation of the criterion.


Criterion Wrappers
------------------

As the DataLoader may automatically cast to the wrong type,
some loss functions need to be wrapped to cast the model outputs and targets to the correct types.
For more information see `this discussion <https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717>`_.

`autrainer` provides the following wrappers:

.. autoclass:: autrainer.criterions.CrossEntropyLoss
   :members:

.. autoclass:: autrainer.criterions.BalancedCrossEntropyLoss
   :members:

.. autoclass:: autrainer.criterions.BalancedBCEWithLogitsLoss
   :members:

.. autoclass:: autrainer.criterions.MSELoss
   :members:

Torch Criterions
----------------

Torch criterions, such as :class:`torch.nn.BCEWithLogitsLoss` for multi-label classification tasks,
can be specified using :ref:`shorthand_syntax` in the dataset configuration, analogous to the criterion wrappers.

.. note::
   It may be necessary to wrap criterions similar to :class:`autrainer.criterions.CrossEntropyLoss`
   or :class:`autrainer.criterions.MSELoss` to cast the model outputs and targets to the correct types.


.. _criterion_setup:

Criterion Setup
---------------

Criterions can optionally provide a :meth:`setup` method which is called after the criterion is initialized
and takes the dataset instance as an argument.
This can be used to set up additional parameters, such as class weights for imbalanced datasets.

.. literalinclude:: ../examples/criterion_setup.py
   :language: python
   :caption: example_loss.ExampleLoss
   :linenos:
   :lines: 10-
.. _loggers:

Loggers
=======

Loggers can be added by specifying a list of :attr:`loggers` in the :ref:`main configuration <main_configuration>` using :ref:`shorthand_syntax`.

.. tip::

   To create custom loggers, refer to the :ref:`custom loggers tutorial <tut_loggers>`.

For example, to add a :class:`MLFlowLogger`, specify it in the :ref:`main configuration <main_configuration>` by adding a list of :attr:`loggers`:

.. code-block:: yaml
   :caption: conf/config.yaml
   :linenos:

   ...
   loggers:
     - autrainer.loggers.MLFlowLogger:
         output_dir: ${results_dir}/.mlflowruns
   ...

Abstract Logger
---------------

All loggers inherit from the :class:`AbstractLogger` class.

.. autoclass:: autrainer.loggers.AbstractLogger
   :members:


Optional Loggers
----------------

The :class:`MLFlowLogger` logs data to MLFlow, while the :class:`TensorBoardLogger` logs data to TensorBoard.
Both loggers require additional dependencies, which are not installed by default.
To install all necessary dependencies, refer to the :ref:`installation` section.

To start the MLflow or TensorBoard server, run the following commands:

.. code-block:: extended

    mlflow server --backend-store-uri /path/to/results/.mlflowruns
    tensorboard --logdir /path/to/results/.tensorboard


In both cases, the path should be the same as the :code:`output_dir` specified in the configuration.

.. autoclass:: autrainer.loggers.MLFlowLogger

.. autoclass:: autrainer.loggers.TensorBoardLogger

Fallback Logger
---------------

If a logger such as :class:`MLFlowLogger` is specified in the configuration,
but the required dependencies are not installed,
the :class:`FallbackLogger` will be used instead.
This logger will log a warning message and will not log any data.

.. autoclass:: autrainer.loggers.FallbackLogger
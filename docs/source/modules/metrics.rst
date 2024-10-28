.. _metrics:

Metrics
=======

Metrics are specified in the :ref:`dataset <datasets>` configuration using :ref:`shorthand syntax <shorthand_syntax>`.
The :attr:`tracking_metric` attribute specifies the metric to be used for early stopping.
The :attr:`metrics` attribute in the dataset configuration specifies a list of metrics to be used during training.

.. tip::

   To create custom metrics, refer to the :ref:`custom metrics tutorial <tut_metrics>`.


For example, a :ref:`dataset <datasets>` for classification could specify the following metrics:

.. literalinclude:: ../examples/metrics.yaml
    :language: yaml
    :caption: conf/dataset/ExampleDataset.yaml
    :linenos:
    :lines: -9

Abstract Metric
---------------

.. autoclass:: autrainer.metrics.AbstractMetric
   :special-members: __call__
   :members:

.. autoclass:: autrainer.metrics.BaseAscendingMetric
   :special-members: starting_metric, suffix

.. autoclass:: autrainer.metrics.BaseDescendingMetric
   :special-members: starting_metric, suffix


Classification Metrics
----------------------

.. autoclass:: autrainer.metrics.Accuracy

.. autoclass:: autrainer.metrics.UAR

.. autoclass:: autrainer.metrics.F1


Multi-label Classification Metrics
----------------------------------

.. autoclass:: autrainer.metrics.MLAccuracy

.. autoclass:: autrainer.metrics.MLF1Macro

.. autoclass:: autrainer.metrics.MLF1Micro

.. autoclass:: autrainer.metrics.MLF1Weighted


(Multi-target) Regression Metrics
---------------------------------

.. autoclass:: autrainer.metrics.CCC

.. autoclass:: autrainer.metrics.MAE

.. autoclass:: autrainer.metrics.MSE

.. autoclass:: autrainer.metrics.PCC
.. _postprocessing:

Postprocessing
==============

.. py:module:: autrainer.postprocessing
   :noindex:

Postprocessing allows for the :ref:`summarization <post_summarization>`, :ref:`aggregation <post_aggregation>`,
as well as  :ref:`grouping <post_grouping>` of the results of the grid search and can be done
using the :ref:`postprocessing CLI <cli_postprocessing>` commands or the
:ref:`postprocessing CLI wrapper <cli_wrapper_postprocessing>` functions.


.. _post_summarization:

Summarization
-------------------

:class:`SummarizeGrid` is used to summarize the results of the grid search.
For each metric, a plot is created.
All validation and test results are stored in a DataFrame.
In addition, a DataFrame summarizing the hyperparameters is created.

.. autoclass:: autrainer.postprocessing.SummarizeGrid
   :members:


.. _post_aggregation:

Aggregation
-------------------

:class:`AggregateGrid` is used to aggregate the results of the grid search.
The results are aggregated over one or more hyperparameters.

.. autoclass:: autrainer.postprocessing.AggregateGrid
   :members:


.. _post_grouping:

Grouping
-------------------

:class:`GroupGrid` is used to manually group the results of the grid search using a Hydra configuration file.
A configuration file is used to define the groups which can be any combination of runs.
The results are grouped according to the configuration file and can span multiple experiments.

The following configuration file illustrates the structure of the configuration file:

.. dropdown:: Manual Grouping Example
   :color: secondary

   Manual grouping is done by defining a YAML configuration file as shown below.
   Multiple experiments (`exp1`, `exp2`, ...) can be created hosting the grouped runs.
   :attr:`runs` is a list of runs that are created for each experiment.

   .. literalinclude:: ../examples/grouping.yaml
      :language: yaml
      :caption: conf/grouping.yaml
      :linenos:


.. autoclass:: autrainer.postprocessing.GroupGrid
   :members:
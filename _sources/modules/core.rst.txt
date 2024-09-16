Core
====

Core provides various utilities and entry points for the `autrainer` framework.

Entry Point
-----------

The main training entry point for `autrainer`.

.. autofunction:: autrainer.main


Instantiation
-------------

Instantiation functions provide wrappers around `Hydra object instantiation <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_,
providing additional type safety and :ref:`shorthand_syntax` support.

.. autofunction:: autrainer.instantiate

.. autofunction:: autrainer.instantiate_shorthand


Utils
-----

Utils provide various helpers for I/O, logging, timing, and hardware information.

.. autoclass:: autrainer.core.utils.Bookkeeping
    :members:

.. autoclass:: autrainer.core.utils.Timer
    :members:

.. autofunction:: autrainer.core.utils.get_hardware_info

.. autofunction:: autrainer.core.utils.save_hardware_info

.. autofunction:: autrainer.core.utils.set_seed

.. autofunction:: autrainer.core.utils.silence


.. _core_plotting:

Plotting
--------

Plotting provides a simple interface to plot metrics of a single run during :ref:`training` as well as multiple runs during :ref:`postprocessing`.

.. tip::

   To create custom plotting configurations, refer to the :ref:`custom plotting configurations tutorial <tut_plotting>`.

By default, training plots are saved as `png` files for each metric.
This can optionally be extended to any format supported by `Matplotlib <https://matplotlib.org/>`_ and additionally pickled for further processing.

.. note::

   Plots are fully customizable by providing `Matplotlib rcParams <https://matplotlib.org/stable/users/explain/customizing.html>`_
   in a :ref:`custom plotting configuration <tut_plotting>`.

.. autoclass:: autrainer.core.plotting.PlotBase
    :members:

.. autoclass:: autrainer.core.plotting.PlotMetrics
    :members:

    .. dropdown:: Default Configurations

       .. configurations::
          :subdir: plotting
          :configs: Default Thesis
          :headline:

Constants
---------

Constants provide various predefined values.

.. autodata:: autrainer.core.constants.NAMING_CONVENTION

.. autodata:: autrainer.core.constants.INVALID_AGGREGATIONS

.. autodata:: autrainer.core.constants.VALID_AGGREGATIONS

.. autodata:: autrainer.core.constants.CONFIG_FOLDERS
.. _schedulers:

Schedulers
==========

Schedulers are optional and by default not used.
This is indicated by the absence of the :attr:`scheduler` attribute in the sweeper configuration (implicitly set to a :attr:`None` configuration file).

In addition to the :attr:`id`, :attr:`_target_`, and scheduler-specific attributes, schedulers can have the following attributes:

* :attr:`step_frequency` (str): The frequency at which the step function is called during training. Possible values are:
  
  * :attr:`batch`: The step function is called after every batch. Defaults to :attr:`batch` if not specified.
  * :attr:`evaluation`: The step function is called after the number of iterations specified by the :attr:`eval_frequency`
    of the :ref:`training configuration <training>`.

To use a scheduler, specify it in the configuration file (:file:`conf/config.yaml`) for the sweeper.

.. tip::

   To create custom schedulers, refer to the :ref:`custom schedulers tutorial <tut_schedulers>`.


.. _torch_schedulers:

Torch Schedulers
----------------

Any scheduler from :class:`torch.optim.lr_scheduler` can be used.
A scheduler is specified in the configuration file as a relative import path for the :attr:`_target_` argument.
Any additional arguments (except the :attr:`id` and :attr:`_target_`) are passed as keyword arguments to the scheduler constructor.

For example, the :class:`torch.optim.lr_scheduler.StepLR` scheduler can be used as follows:

.. configurations::
      :subdir: scheduler
      :configs: StepLR
      :exact:

.. dropdown:: Default Configurations

   **None**

   This configuration file is used to indicate that no scheduler is used and serves as a no-op placeholder.

   .. configurations::
      :subdir: scheduler
      :configs: None
      :exact:
   
   .. configurations::
      :subdir: scheduler
      :configs: StepLR
      :headline:
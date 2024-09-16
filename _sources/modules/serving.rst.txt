Inference
=========

Inference offers an interface to obtain predictions or embeddings from a trained model.
In addition, a sliding window can be used to obtain predictions or embeddings from parts of the input data.

The :ref:`autrainer inference <cli_inference>` CLI command and the :meth:`~autrainer.cli.inference` CLI wrapper function
allow for the (sliding window) inference of audio data using a trained model.

.. note::

   Currently, inference is only supported for audio data.


.. _audio_inference:

Audio Inference
---------------

.. autoclass:: autrainer.serving.Inference
   :members:
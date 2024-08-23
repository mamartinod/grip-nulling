Build a GRIP instrument model
=============================

Rules on the inputs
-------------------

Any instrument model to be read by GRIP must follow the template of the arguments function:

`def func_instru(na, wavelength, wl_idx, *args)`

The first argument must be the self-calibrated null depth,
the second one must be the wavelength as a scalar,
the third one must be an index to pick the correct value of the spectral quantities we may call in `*args`.
Then the sequence of arguments needed to run the model.

Rule on the outputs
-------------------

The output of the function must be a tuple.
GRIP allows to track what is happening inside the instrument model for diagnostic purpose.
Thus the function above must return a tuple or equivalent, even if there is a single quantity to return.

The use of Cupy
---------------

GRIP uses the GPU with the Cupy library.
In case it is not installed, numpy is used instead the switch from the former to the latter is very drafty:

.. code-block:: python

	try:
	   import cupy as cp
	except ModuleNotFoundError:
	   import numpy as cp





GRIP puts everything relevant on the GPU for fast computations (this model and the generation of the simulated histograms).
Unless the workstation does not have cupy installed, any numpy functions used inside the model must be written with their cupy equivalent and the float defined in 32 bits (faster computations according to Nvidia's GPU specs, unless the machine has the top-notch of their catalog for AI and big data).

Using Jax would naturally solve this issue and it should be use in a future release.

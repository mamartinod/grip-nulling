.. grip documentation master file, created by
   sphinx-quickstart on Fri Jan 19 15:50:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================
GRIP documentation
==================

What is GRIP?
=============

GRIP (Generic data Reduction for nulling Interferometry Package) is a toolbox for reducing nulling data
with the nulling self-calibration method (NSC).
These tools can work with data coming from any nuller nuller.
It handles baseline discrimination, spectral dispersion.
GRIP currently models the histogram of the data in order to get:
- the self-calibrated null depth
- the mean and standard deviation of a normally distributed observable (e.g. OPD)

GRIP currently features several optimizing strategy:
- least squares
- maximum likelihood
- MCMC (with the emcee library)

It can work on GPU thanks to the cupy library but it does not handle Jax yet.

GRIP is open-source and can be found on `Github <https://github.com/mamartinod/grip>`_.

Dependencies
============
- numpy >= 1.26.2
- scipy >= 1.11.4
- matplotlib >= 3.6.3
- h5py >= 3.8.0
- emcee >= 3.1.4
- numdifftools >= 0.9.41
- astropy >= 5.2.1
- cupy >= 11.5.0 (optional and not downloaded during the installation)
- sbi >= 0.23.2 (optional and not downloaded during the installation)
- pytorch >= 2.1.2 (optional and not downloaded during the installation)

Installation
============

From PIP
--------
Use the command ``pip install grip-nulling``.

To uninstall: ``pip uninstall grip-nulling``.

From the source
---------------
1. Clone, download the repo or check one of the releases.
2. Open the directory then a terminal
3. Use the command ``pip install .`` or ``conda install .``

To uninstall:

1. Open a terminal and the environment
2. Do not locate yourself in the directory of the package or the parent
3. Type ``pip uninstall grip``
4. Delete the directory ``grip``

GPU powering
============
If you have a GPU, greatly boost the performance of GRIP by using `Cupy <https://cupy.dev/>`_.

Using Neural Posterior Estimation
=================================
To use the Neural Posterior Estimation technique, the libraries `SBI <https://github.com/sbi-dev/sbi>`_ 
and `PyTorch <https://pytorch.org/>`_ must be installed separately.

GPU is not necessary to use the NPE feature of GRIP.

Tutorials
=========
Tutorials are available on the `Github <https://github.com/mamartinod/grip-nulling>`__ page of the project.


Navigation
==========
.. toctree::
   :maxdepth: 1

   grip_nested_architecture
   build_model
   fitting
   generic
   histogram_tools
   instrument_models
   load_files
   npe   
   preprocessing
   plots

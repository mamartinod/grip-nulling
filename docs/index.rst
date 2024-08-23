.. grip documentation master file, created by
   sphinx-quickstart on Fri Jan 19 15:50:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GRIP documentation
==================

What is GRIP?
-------------

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

Dependencies
------------
- numpy >= 1.26.2
- scipy >= 1.11.4
- matplotlib >= 3.6.3
- h5py >= 3.8.0
- emcee >= 3.1.4
- cupy >= 11.5.0 (optional and not downloaded during the installation)

Installation
------------
1. Clone or download the repo.
2. Open the folder then a terminal
3. Use the command ``pip install .`` or ``conda install .``.

Tutorials
---------
See the main page of the project on Github.

Future work
-----------
Contributions are welcome.

- Build a double-Bracewell model
- Port it to Jax
- Add machine learning techniques
- Extend the usecase to interferometry
- Extend the capability to fit an arbitrary number of parameters
- Design a logo


Navigation
==========
.. toctree::
   :maxdepth: 2

   grip_nested_architecture
   build_model
   preprocessing
   plots
   instrument_models
   histogram_tools
   generic
   fitting
   load_files


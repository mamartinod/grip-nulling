# Generic data Reduction for nulling Interferometry Package
Reaching extreme interferometric contrasts relies as much on the hardware as on the data processing technique, which is one of the main research pillars of SCIFY. 
Over the past decade, self-calibration data reduction techniques have been developed and proven to improve the final contrast after post-processing by a factor of at least 10 over classical reduction techniques (Hanot et al. 2011, Mennesson et al. 2011, Defrère et al. 2016, Mennesson et al. 2016, Norris et al. 2020, Martinod et al. 2021). 
Over the year, several nulling self-calibration pipelines have been written. Within SCIFY, the goals are:
1. to develop a generic nulling self-calibration pipeline with all state-of-the-art features of high-contrast nulling data reduction and validate it on existing nulling data obtained with the LBTI survey; 
2. to primarly focus on the use case of NOTT
3. to improve the versatility and performance of the pipeline by adding dispersed modes and better ways to compute the error bars (e.g., MCMC);
4. to make this software open-source so that it can serve the whole community and serve as a basis for future developments


# Documentation
Find the documentation [here](https://mamartinod.github.io/grip-nulling/).

For the documentation of specific releases, see the [ReadTheDocs](https://grip.readthedocs.io/en/stable/).

# Tutorials
1. [How to get the histograms of the data and the models](/tutorials/tuto1_get_histo_and_display.ipynb)
2. [How to scan the parameter space with a binomial likelihood estimator](/tutorials/tuto2_explore_parameter_space.ipynb)
3. [How to perform a fit with a binomial likelihood estimator](/tutorials/tuto3_fit_with_likelihood.ipynb)
4. [How to use a MCMC approach](/tutorials/tuto4_use_of_mcmc.ipynb)
5. [How to build your own model of the instrument](/tutorials/tuto5_build_your_own_model.ipynb)
6. [Use Neural Posterior Estimation for a fast and amortized inference](/tutorials/tuto6_npe.ipynb)

# Installation
## Dependencies
- numpy >= 1.26.2
- scipy >= 1.11.4
- matplotlib >= 3.6.3
- h5py >= 3.8.0
- emcee >= 3.1.4
- numdifftools >= 0.9.41
- astropy >= 5.2.1
- cupy >= 11.5.0 (optional and not downloaded during the installation)
- lampe >= 0.8.2 (optional and not downloaded during the installation)
- pytorch >= 2.1.2 (optional and not downloaded during the installation)

## From PIP
1. Use the command ``pip install grip-nulling``

To uninstall: ``pip uninstall grip-nulling``

## From the repo
1. Clone, download the repo or check one of the releases.
2. Open the directory then a terminal
3. Use the command ``pip install .`` or ``conda install .``
4. Visit the documentation and its tutorial to discover more about the library

To uninstall:
1. Open a terminal and the environment
2. Do not locate yourself in the folder of the package or the parent
3. Type `pip uninstall grip`
4. Delete the directory `grip`

# GPU powering
If you have a GPU, greatly boost the performance of GRIP by using `Cupy <https://cupy.dev/>`_.

# Using Neural Posterior Estimation
To use the Neural Posterior Estimation technique, the libraries [LAMPE](https://github.com/probabilists/lampe) 
and [PyTorch](https://pytorch.org/) must be installed separately.

GPU is not necessary to use the NPE feature of GRIP.

# Citation
Please cite [Martinod et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024SPIE13095E..1AM/abstract) whenever you publish data reduced with GRIP and the relevant publication(s) for the algorithms you use within GRIP.
They are usually mentioned in the documentation.

The paper is also on [Arxiv](https://arxiv.org/abs/2407.08802).

# Acknowledgements
GRIP is a development carried out in the context of the [SCIFY project](http://denis-defrere.com/scify.php). SCIFY has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No 8660).

The documentation of the software package is funded by the European Union's Horizon 2020 research and innovation program under grant agreement No. 101004719.

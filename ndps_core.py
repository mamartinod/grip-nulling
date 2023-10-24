# -*- coding: utf-8 -*-
"""
Work in progress:
    - rethink spectral binning, inconsistencies in the code: binned wl_scale is never used in the nsc_function, accum shape is not good nor properly filled
    - ``activate_use_of_photometry'': to put in the instrument model?
    - scattered parameters related to data between config and run (e.g. wavelength)
"""
"""
This code determines the astrophysical null depth either by the NSC method
(https://ui.adsabs.harvard.edu/abs/2011ApJ...729..110H/abstract)
which does not necessarily requires a calibrator, either by a classic
measurement (https://ui.adsabs.harvard.edu/abs/2011ApJ...729..110H/abstract)
which must be calibrated.

The fit is done via the fitting algorithm Trust Region Reflective used in the
scipy function least_squares
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
In order to be more resilient to local minima, basin hopping is practiced 10
times by slighly changing the initial conditions (see the documentation of the
function ``basin_hoppin_values`` below).

NOTE: the code is not fully stabilised yet so some discarded functions exist
but are not used
anymore. Similarly, the execution is highly customisable due to the different
explorations done.

The data to load and the settings about the space parameters are handled by the
script ``nsc_config.py``.
This script handles the way the null depth is determined and on which baseline.
The library is ``nsc_lib.py``. However, some core functions
are coded here:

        * **gaussian**: to generate Gaussian curves;
        * **nsc_function**: the core function to create and fit the histogram\
            of synthetic null depths to the real histograms;
        * **Logger**: class logging all the console outputs into a log file;
        * **basin_hoppin_values**: function changing initial condition to do\
            basin hopping on the fitting algorithm.

Fitted values are stored into npy file (one per basin iteration in case of\
                                        breakdown of the program).
Pickle files stored the results from all iterations.
Plots of the fit of the histogram and plots of the intermediate quantities
(histogram of (anti-)null outputs, synthetic or real, photometries, injection)
are also saved.

The settings of the script are defined by:

        * **activate_linear_model**: bool,
                use the linear model :math:`N = Na + Ninstr`;
        * **activate_use_antinull**: bool. If ``True``,
                                    it use the antinull output;
        * **activate_spectral_sorting**: bool. If ``True``,it keeps frames for
                which the null depths are between boundaries for every spectral
                channels;
        * **activate_random_init_guesses**: bool. If ``True``, it activates the
                basin hopping. It is automatically set to ``True`` after the
                first iteration;
        * **activate_spectral_binning**: bool. If ``True``, it bins the
                spectral channels;
        * **activate_use_photometry**: bool. If ``True``, it uses the
                photometric outputs instead of simulating sequences of
                photometries for the fit. It is advised to keep it at ``False``
                as its use gives non-convincing results and more investigation
                are required;
        * **activate_dark_correction**: bool. If ``True``, it corrects the
                distribution of the injection from the contribution of the
                detector noise assuming they all follow normal distribution.
                It is adviced to keep it at ``False``;
        * **activate_save_classic_esti**: bool. If ``True``, it gives a broadband
                null depth to calibrate in the old-school way;
        * **activate_phase_sorting**: bool. If ``True``, it keeps the frame for
                which the OPD follows a normal distribution (anti-LWE filter).
                It is strongly recommended to keep it ``True``.
        * **activate_preview_only**: bool. If ``True``, the script stops after
                sorting the frame according to the phase distribution and plot
                the sequences of frames with highlighting the kept ones to tune
                the parameters of the filter;
        * **nb_frames_sorting_binning**: integer, number of frames to bin to
                set the frame sorting parameters. Typical value is 100;
        * **activate_time_binning_photometry**: bool. If ``True``, it bins the
                photometric outputs temporally. It is strongly recommended to
                keep it ``True`` to mitigate the contribution of the detector
                noise;
        * **nb_frames_binning_photometry**: integer, number of frames to bin
                the photometric outputs. Typical value is 100;
        * **global_binning**: integer, frames to bin before anything starts in
                order to increase SNR;
        * **wl_min**: float, lower bound of the bandwidth to process;
        * **wl_max**: float, upper bound of the bandwidth to process;
        * **n_samp_total**: int, number of total Monte-Carlo values to
                generate. Typical value is 1e+8;
        * **n_samp_per_loop**: int, number of Monte-Carlo values to generate
                per loop (to not saturate the memory). Typical value is 1e+7;
        * **activate_zeta**: bool. If ``True``, it uses the measured zeta
                coeff. If False, value are set to 0.5;
        * **activate_oversampling**: bool. If ``True``, it includes the loss of
                coherence due to non-monochromatic spectral channels;
        * **skip_fit**: bool. If ``True``, the histograms of data and model
                given the initial guesses are plotted without doing the fit.
                Useful to just display the histograms and tune the settings;
        * **chi2_map_switch**: bool. If ``True``, it grids the parameter space
                in order to set the boundaries for the fitting algorithm;
        * **nb_files_data**: 2-int tuple. Load the data files comprised between
                the values in the tuple;
        * **nb_files_dark**: 2-int tuple. Load the dark files comprised between
                the values in the tuple;
        * **basin_hopping_nloop**: 2-int tuple. Lower and upper bound of the
                iteration loop for basin hopping method;
        * **which_nulls**: list. List of baselines to process;
        * **map_na_sz**: int. Number of values of astrophysical null depth in
                the grid of the parameter space;
        * **map_mu_sz**: int. Number of values of :math:`\mu` in the grid of
                the parameter space;
        * **map_sig_sz**: int. Number of values of :math:`\sig` in the grid of
                the parameter space.
        * **activate_photo_resampling**: Deconvolve photometries after
                assuming they follow a Normal distribution and the values of
                the photometries are replaced by RV coming from
                that distribution.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from timeit import default_timer as time
import os
import sys
import ndps_lib as gff
import pickle
from datetime import datetime
from ndps_config import prepareConfig
from cupyx.scipy.special import ndtr
import corner

plt.ioff()

def gaussian(x, A, loc, sig):
    """
    Computes a Gaussian curve

    :Parameters:

        **x**: values where the curve is estimated.

        **A**: amplitude of the Gaussian.

        **x0**: location of the Gaussian.

        **sig**: scale of the Gaussian.

    :Returns:

        Gaussian curve.
    """
    return A * np.exp(-(x-loc)**2/(2*sig**2))

def nsc_function(bins0, na, mu_opd, sig_opd, *args, **kwargs):
    """Do Monte-Carlo for NSC method.

    It fits polychromatic data to give one achromatic null depth.
    This functions runs on the GPU with cupy (https://cupy.dev/).

    The arguments of the function are the parameters to fit.
    The other values required to calculated the synthetic sequences of null
    depth are imported
    via global variables listed at the beginning of the function.

    The verbose of this function displays the input parameters to track the
    convergence of the fitting algorithm.

    Global variables include quantities necessary to fit the histograms and
    intermediate products to control the efficiency of the fit.

    The functions loops over the spectral channels to create independent
    histograms for each of them and loops over the Monte-Carlo samples to avoid
    the saturation of the memory.
    For the latter, the histograms are added together then averaged to get the
    normalization to 1.

    :Parameters:

        **bins0**: array
            Bins of the histogram of the null depth to fit

        **na**: float
            Astrophysical null depth, quantity of interest.

        **mu_opd**: float
            Instrumental OPD of the considered baseline.

        **sig_opd**: float
            Sigma of the normal distribution describing the fluctuations of
            the OPD.

        **args**: optional
            Array, use to fit the parameters ``mu`` and ``sig`` of the normal
            distribution of Ir
            (https://ui.adsabs.harvard.edu/abs/2011ApJ...729..110H/abstract)
            if the antinull output is not used. arg can be either a 2-element
            or a :math:`2 \times number of spectral channels`-element array,
            depending on the fit of a unique Ir or one per spectral channel.

    :Returns:

        **accum**: array
            Histogram of the synthetic sequences of null depth.
    """
    # Imported in the funcion
    global data_IA_axis, cdf_data_IA, data_IB_axis, cdf_data_IB, spectra  # GPU
    global zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B
    global data_IA, data_IB
    global dark_Iminus_cdf, dark_Iminus_axis  # On GPU
    global dark_Iplus_cdf, dark_Iplus_axis  # On GPU
    global spec_chan_width
    global activate_use_photometry, activate_linear_model
    global activate_use_antinull, activate_spectral_binning, activate_lbti_mode
    global sigma_eps_cdf, sigma_eps_axis # On GPU
    global std_dark_Iminus
    global rvu_opd, rvu_sigmaeps, rvu_injectionA, rvu_injectionB, rvu_darkm, rvu_darkp

    # generated by the function
    global rv_IA, rv_IB, rv_opd, rv_dark_Iminus, rv_dark_Iplus  # On GPU
    global rv_null, rv_interfminus, rv_interfplus  # On GPU
    global interfminus, interfplus  # On GPU
    global phase_bias, bins, dphase_bias
    global interfminus_binned, interfplus_binned
    global rv_sigma_eps

    # Parameters of the MC process
    global n_samp_per_loop, count, wl_scale0, nloop, number_of_Ir
    global activate_oversampling, switch_invert_null
    global rv_IA_list

    # Test and diagnostic
    global Iplus, liste_rv_interfminus, liste_rv_interfplus
    global liste_rv_dark_Iminus, liste_rv_dark_Iplus
    global liste_rv_IA, liste_rv_IB, rv_injectionA, rv_injectionB
    global injection_mean, injection_corrected_std

    if 'normed' in kwargs.keys() and kwargs['normed'] == False:
        normed = False
    else:
        normed = True


    try:    
        count += 1
    except NameError:
        count = 0
    
    if 'verbose' in kwargs.keys() and kwargs['verbose'] == False:
        pass
    else:
        text_intput = (int(count), na, mu_opd, sig_opd, *args)
        print(text_intput)

    # Axes: spectral channel, number of bins
    accum = cp.zeros((bins0.shape[0], bins0.shape[1]-1), dtype=cp.float32)

    if wl_scale0.size > 1:
        spec_chan_width = abs(np.mean(np.diff(wl_scale0)))
    else:
        spec_chan_width = 5

    if activate_lbti_mode == True:
        rv_sigma_eps = gff.rv_generator(sigma_eps_axis, sigma_eps_cdf, n_samp_per_loop, rvu_sigmaeps)
        rv_sigma_eps = rv_sigma_eps.astype(cp.float32)
        
    opd_axis = cp.linspace(mu_opd - 6 * sig_opd, mu_opd + 6 * sig_opd, 1001, dtype=cp.float32)
    opd_cdf = ndtr((opd_axis - mu_opd) / sig_opd)
    rv_opd = gff.rv_generator(opd_axis, opd_cdf, n_samp_per_loop, rvu_opd)
    rv_opd = rv_opd.astype(cp.float32)
    
    # Generate random values of injection
    if not activate_use_photometry:
        # random values for photometry A
        rv_injectionA = gff.rv_generator(
            data_IA_axis, cdf_data_IA, n_samp_per_loop, rvu_injectionA)
        # random values for photometry B
        rv_injectionB = gff.rv_generator(
            data_IB_axis, cdf_data_IB, n_samp_per_loop, rvu_injectionB)

    '''
    Number of samples to simulate is high and the memory is low so we iterate
    to create an average histogram
    '''
    for _ in range(nloop):
        liste_rv_interfminus = cp.zeros(
            (wl_scale0.size, n_samp_per_loop), dtype=cp.float32)
        liste_rv_interfplus = cp.zeros(
            (wl_scale0.size, n_samp_per_loop), dtype=cp.float32)
        liste_rv_dark_Iminus = cp.zeros(
            (wl_scale0.size, n_samp_per_loop), dtype=cp.float32)
        liste_rv_dark_Iplus = cp.zeros(
            (wl_scale0.size, n_samp_per_loop), dtype=cp.float32)
        liste_rv_IA = cp.zeros(
            (wl_scale0.size, n_samp_per_loop), dtype=cp.float32)
        liste_rv_IB = cp.zeros(
            (wl_scale0.size, n_samp_per_loop), dtype=cp.float32)


        if activate_spectral_binning:
            interfminus_binned = cp.zeros(n_samp_per_loop, dtype=cp.float32)
            interfplus_binned = cp.zeros(n_samp_per_loop, dtype=cp.float32)


        for k in range(wl_scale0.size):  # Iterate over the wavelength axis
            # random values for dark noise
            rv_dark_Iminus = gff.rv_generator(
                dark_Iminus_axis[k], dark_Iminus_cdf[k], n_samp_per_loop, rvu_darkm[k])
            rv_dark_Iminus = rv_dark_Iminus.astype(cp.float32)

            rv_dark_Iplus = gff.rv_generator(
                dark_Iplus_axis[k], dark_Iplus_cdf[k], n_samp_per_loop, rvu_darkp[k])
            rv_dark_Iplus = rv_dark_Iplus.astype(cp.float32)


            if not activate_use_photometry:
                rv_IA = rv_injectionA * spectra[0][k]
                rv_IB = rv_injectionB * spectra[1][k]
                
            else:
                rv_IA = cp.asarray(data_IA, dtype=cp.float32)
                rv_IB = cp.asarray(data_IB, dtype=cp.float32)


            if activate_lbti_mode == True:
                rv_null, rv_interfminus, rv_interfplus =\
                    gff.computeNullDepthLBTI(na, rv_IA, rv_IB, wl_scale0[k],
                                          rv_opd, np.pi, dphase_bias,
                                          rv_dark_Iminus, 
                                          zeta_minus_A[k], zeta_minus_B[k],
                                          zeta_plus_A[k], zeta_plus_B[k],
                                          2600,
                                          activate_oversampling, rv_sigma_eps)
            else:
                rv_null, rv_interfminus, rv_interfplus =\
                    gff.computeNullDepth(na, rv_IA, rv_IB, wl_scale0[k],
                                          rv_opd, phase_bias, dphase_bias,
                                          rv_dark_Iminus, rv_dark_Iplus,
                                          zeta_minus_A[k], zeta_minus_B[k],
                                          zeta_plus_A[k], zeta_plus_B[k],
                                          spec_chan_width,
                                          activate_oversampling,
                                          switch_invert_null)                

            if activate_spectral_binning:
                interfminus_binned += rv_interfminus
                interfplus_binned += rv_interfplus

            if switch_invert_null:
                rv_null = rv_interfplus / rv_interfminus
            else:
                rv_null = rv_interfminus / rv_interfplus

            rv_null = rv_null[~np.isnan(rv_null)]  # Remove NaNs
            rv_null = cp.sort(rv_null)

            
            bins = cp.asarray(bins0[k], dtype=cp.float32)
            pdf_null = cp.histogram(rv_null, bins)[0]

            if normed:
                accum[k] += pdf_null / cp.sum(pdf_null)
            else:
                accum[k] += pdf_null

            """
            Save some debug stuff
            """
            liste_rv_interfminus[k] = rv_interfminus
            liste_rv_interfplus[k] = rv_interfplus
            liste_rv_dark_Iminus[k] = rv_dark_Iminus
            liste_rv_dark_Iplus[k] = rv_dark_Iplus
            liste_rv_IA[k] = rv_IA
            liste_rv_IB[k] = rv_IB
            
            """
            End of loop on spectral channels
            """

        """
        Inverting the fake sequences if the nll and antinull outputs are
        swaped.
        Non-spectrally binned are already swaped in the "computeNullDepth"
        function above
        """
        if activate_spectral_binning:
            if switch_invert_null:
                rv_null = interfplus_binned / interfminus_binned
            else:
                rv_null = interfminus_binned / interfplus_binned

            rv_null = rv_null[~np.isnan(rv_null)]  # Remove NaNs
            rv_null = cp.sort(rv_null)
            pdf_null = cp.histogram(rv_null, cp.asarray(
                bins0[0], dtype=cp.float32))[0]
            if normed:
                accum[0] += pdf_null / cp.sum(pdf_null)
            else:
                accum[0] += pdf_null / rv_null.size

    # Compute the average histogram over the nloops iterations
    accum = accum / nloop
    if cp.all(cp.isnan(accum)):
        accum[:] = 0
    accum = cp.asnumpy(accum)

    return accum.ravel()


def basin_hoppin_values(mu_opd0, sig_opd0, na0, bounds_mu, bounds_sig,
                        bounds_na):
    """Create several initial guesses.

    Create as many as initial guess as there are basin hopping iterations
    to do.
    The generation of these values are done with a normal distribution.

    :param mu_opd0: Instrumental OPD around which random initial guesses are
                    created.
    :type mu_opd0: float
    :param sig_opd0: Instrumental OPD around which random initial guesses are
                    created.
    :type sig_opd0: float
    :param na0: Null depth of the source around which random initial guesses
                are created.
    :type na0: float
    :param bounds_mu: Lower and upper bounds between which the random values of
                    **mu_opd** must be.
    :type bounds_mu: 2-tuple
    :param bounds_sig: Lower and upper bounds between which the random values
                    of **sig_opd** must be.
    :type bounds_sig: 2-tuple
    :param bounds_na: Lower and upper bounds between which the random values of
                    **na** must be.
    :type bounds_na: 2-tuple
    :return: New initial guess for **mu_opd**,
            new initial guess for **sig_opd**,
            new initial guess for **na**.
    :rtype: 3-tuple

    """
    print('Random withdrawing of init guesses')

    for _ in range(1000):
        mu_opd = np.random.normal(mu_opd0, 50)
        if mu_opd > bounds_mu[0] and mu_opd < bounds_mu[1]:
            break
        if _ == 1000-1:
            print('mu_opd: no new guess, take initial one')
            mu_opd = mu_opd0

    for _ in range(1000):
        sig_opd = abs(np.random.normal(sig_opd0, 50.))
        if sig_opd > bounds_sig[0] and sig_opd < bounds_sig[1]:
            break
        if _ == 1000-1:
            print('sig opd: no new guess, take initial one')
            sig_opd = sig_opd0

    for _ in range(1000):
        na = np.random.normal(na0, 0.03)
        if na > bounds_na[0] and na < bounds_na[1]:
            break
        if _ == 1000-1:
            print('na: no new guess, take initial one')
            na = na0

    print('Random drawing done')
#    np.random.set_state(orig_seed)
    return mu_opd, sig_opd, na


class Logger(object):
    """Save the content of the console inside a txt file.

    Class allowing to save the content of the console inside a txt file.
    Found on the internet, source lost.
    """

    def __init__(self, log_path):
        """Init instance of the class.

        :param log_path: path to the log file.
        :type log_path: str

        """
        self.orig_stdout = sys.stdout
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        """Print the content in the terminal and in the log file.

        :param message: message to print and log
        :type message: str

        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Present for python 3 compatibility.

        This flush method is needed for python 3 compatibility.
        This handles the flush command by doing nothing.
        You might want to specify some extra behavior here.
        """
        pass

    def close(self):
        """Close the log file.

        Close the log file and print in the terminal is back to default
        settings.
        """
        sys.stdout = self.orig_stdout
        self.log.close()
        print('Stdout closed')


def run_ndps(activates, skip_fit, chi2_map_switch, maps_sz, nbs, which_nulls,
            wl_minmax, supercount, resuts_rsc):
    """Run the NSC script to fit the self-calibrated null depth.

    :param activates: pack all the booleans to tune the script
    :type activates: tuple
    :param skip_fit: Skip the fit and just display the histograms and other
        quantities if ``True``
    :type skip_fit: bool
    :param chi2_map_switch: Map the parameter spaces instead of doing the fit
        if ``True``.
    :type chi2_map_switch: bool
    :param maps_sz: size of the ranges of Na, mu_opd, sig_opd, resp.
    :type maps_sz: tuple
    :param nbs: pack all the numerical values tuning the script.
    :type nbs: tuple
    :param which_nulls: list of baselines to process
    :type which_nulls: list
    :param wl_minmax: min and max of the spectral range to process, in nm
    :type wl_minmax: tuple
    :raises UserWarning: raised if the list of data to process is empty.
    :raises Exception: raised if initial guess from config file are out of
        the boundaries for any of the 6 baselines.

    """
    ''' GLobal variables used and calls in *nsc_function*. '''
    # Imported in the funcion
    global data_IA_axis, cdf_data_IA, data_IB_axis, cdf_data_IB, spectra  # GPU
    global zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B
    global data_IA, data_IB
    global dark_Iminus_cdf, dark_Iminus_axis  # On GPU
    global dark_Iplus_cdf, dark_Iplus_axis  # On GPU
    global spec_chan_width
    global activate_use_photometry, activate_linear_model
    global activate_use_antinull
    global sigma_eps_cdf, sigma_eps_axis # On GPU
    global std_dark_Iminus
    global rvu_opd, rvu_sigmaeps, rvu_injectionA, rvu_injectionB, rvu_darkm, rvu_darkp
    

    # generated by the function
    global rv_IA, rv_IB, rv_opd, rv_dark_Iminus, rv_dark_Iplus  # On GPU
    global rv_null, rv_interfminus, rv_interfplus  # On GPU
    global interfminus, interfplus  # On GPU
    global phase_bias, bins, dphase_bias
    global interfminus_binned, interfplus_binned

    # Parameters of the MC process
    global n_samp_per_loop, count, wl_scale0, nloop, number_of_Ir
    global activate_oversampling, switch_invert_null
    global rv_IA_list

    # Test and diagnostic
    global Iplus, liste_rv_interfminus, liste_rv_interfplus
    global liste_rv_IA, liste_rv_IB, rv_injectionA, rv_injectionB
    global liste_rv_dark_Iminus, liste_rv_dark_Iplus
    global injection_mean, injection_corrected_std
    
    # Activate variables
    global activate_dark_correction, activate_phase_sorting,\
        activate_linear_model,\
        activate_oversampling, activate_preview_only,\
        activate_random_init_guesses, activate_photo_resampling,\
        activate_save_classic_esti, activate_spectral_sorting,\
        activate_spectral_binning, activate_time_binning_photometry,\
        activate_use_antinull, activate_use_photometry,\
        activate_zeta, activate_lbti_mode

    activate_dark_correction, activate_phase_sorting, activate_linear_model,\
        activate_oversampling, activate_preview_only,\
        activate_random_init_guesses, activate_photo_resampling,\
        activate_save_classic_esti, activate_spectral_sorting,\
        activate_spectral_binning, activate_time_binning_photometry,\
        activate_use_antinull, activate_use_photometry,\
        activate_zeta, activate_remove_dark, activate_draw_model, activate_lbti_mode,\
        select_optimizer, activate_rvu, activate_mcmc = activates

    map_na_sz, map_mu_sz, map_sig_sz = maps_sz
    global_binning, n_samp_total, n_samp_per_loop, nb_frames_sorting_binning,\
        nb_frames_binning_photometry, nb_files_data, nb_files_dark,\
        basin_hopping_nloop = nbs
        
    if select_optimizer == 0:
        normed = True
        label_optimizer = 'chi2'
    else:
        normed = False
        label_optimizer = 'lklh'

    wl_min, wl_max = wl_minmax

    nloop = n_samp_total // n_samp_per_loop

    # Load the custom configuration for the dataset
    config = prepareConfig(supercount)
    # If one null and antinull outputs are swapped in the data processing
    nulls_to_invert = config['nulls_to_invert']
    # If one null and antinull outputs are swapped in the data processing
    nulls_to_invert_model = config['nulls_to_invert_model']

    '''
    Set the bounds of the parameters to fit
    '''
    # bounds for DeltaPhi mu, one tuple per null
    bounds_mu0 = config['bounds_mu0']
    # bounds for DeltaPhi sig
    bounds_sig0 = config['bounds_sig0']
    bounds_na0 = config['bounds_na0']  # bounds for astronull

    '''
    Used for computing the finite difference in the TRF fitting algorithm
    scale factor of the parameters to fit, see least_squares doc for more
    details.
    '''
    diffstep = config['diffstep']
    xscale = config['xscale']
    # Boundaries of the histogram
    bin_bounds0 = config['bin_bounds0']

    '''
    Set the initial conditions
    '''
    # initial guess of DeltaPhi mu for the 6 baselines
    mu_opd0 = config['mu_opd0']
    # initial guess of DeltaPhi sig for the 6 baselines
    sig_opd0 = config['sig_opd0']
    na0 = config['na0']  # initial guess of astro null for the 6 baselines

    """ Import real data """
    datafolder = config['datafolder']
    darkfolder = config['darkfolder']
    datafolder = datafolder
    darkfolder = darkfolder
    file_path = config['file_path']
    save_path = config['save_path']
    data_list = config['data_list'][nb_files_data[0]:nb_files_data[1]]
    dark_list = config['dark_list'][nb_files_dark[0]:nb_files_dark[1]]
    zeta_coeff_path = config['zeta_coeff_path']
    factor_minus0, factor_plus0 =\
        config['factor_minus0'], config['factor_plus0']

    print('Loaded configuration for %s, at the date %s' %
          (config['starname'], config['date']))

    if len(data_list) == 0 or len(dark_list) == 0:
        raise UserWarning('data list or dark list is empty')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """
    Some constants/useless variables kept for retrocompatibility
    """
    dphase_bias = 0.  # constant value for corrective phase term
    phase_bias = 0.  # Phase of the fringes

    # Indexes of null, beam A and beam B, zeta label for antinull, segment id
    null_table = {'null1': [0, [0, 1], 'null7'],
                  'null2': [1, [1, 2], 'null8'],
                  'null3': [2, [0, 3], 'null9'],
                  'null4': [3, [2, 3], 'null10'],
                  'null5': [4, [2, 0], 'null11'],
                  'null6': [5, [3, 1], 'null12']}

    # Specific settings for mapping parameters space
    if chi2_map_switch:
        n_samp_per_loop = int(1e+5)  # number of samples per loop
        nloop = 1
        basin_hopping_nloop = (
            basin_hopping_nloop[0], basin_hopping_nloop[0]+1)

    # Specific settings for non-fitting
    if skip_fit or activate_mcmc:
        basin_hopping_nloop = (
            basin_hopping_nloop[0], basin_hopping_nloop[0]+1)

    # Fool-proof
    if not chi2_map_switch and not skip_fit:
        check_mu = gff.check_init_guess(mu_opd0, bounds_mu0, bounds_mu0)
        check_sig = gff.check_init_guess(sig_opd0, bounds_sig0, bounds_sig0)
        check_null = gff.check_init_guess(na0, bounds_na0, bounds_na0)

        if check_mu or check_sig or check_null:
            raise Exception('Check boundaries: the initial guesses' +
                            '(marked as True) are not between the boundaries' +
                            '(null:%s, mu:%s, sig:%s).' % (check_null,
                                                           check_mu,
                                                           check_sig))

    total_time_start = time()
    for key in which_nulls:  # Iterate over the null to fit
        # =====================================================================
        # Import data
        # =====================================================================
        print('****************')
        print('Processing %s' % key)
        if activate_preview_only:
            msg = 'Preview only'
        elif activate_save_classic_esti:
            msg = 'Classic estimation'
        elif skip_fit and not activate_preview_only and not chi2_map_switch:
            msg = 'Skip fit'
        elif not skip_fit and not activate_preview_only and not chi2_map_switch:
            msg = 'NSC'
        elif chi2_map_switch and not activate_preview_only:
            msg = label_optimizer+'map scanning'
        elif activate_mcmc:
            msg = 'MCMC'
        else:
            msg = 'Unknown'
        
        print('Execution mode: ' + msg)
     
    #    plt.ioff()
        start_loading = time()
        if key in nulls_to_invert_model:
            switch_invert_null = True
        else:
            switch_invert_null = False

        # Select the index of the null output to process
        idx_null = null_table[key][0]
        # Select the indexes of the concerned photometries
        idx_photo = null_table[key][1]
        # Select the index of the antinull output to process
        key_antinull = null_table[key][2]

        ''' Load data about the null to fit '''
        dark = gff.load_data(dark_list, (wl_min, wl_max),
                             key, nulls_to_invert,
                             frame_binning=global_binning, lbti=activate_lbti_mode)
        data = gff.load_data(data_list, (wl_min, wl_max), key,
                             nulls_to_invert, dark,
                             frame_binning=global_binning, lbti=activate_lbti_mode)
        stop_loading = time()

        
        if activate_phase_sorting or activate_preview_only:
            data, idx_good_frames, i_pm = gff.sortFrames(
                data, nb_frames_sorting_binning, 0.1, factor_minus0[idx_null],
                factor_plus0[idx_null], key, os.path.basename(datafolder[:-1]),
                plot=activate_preview_only,
                save_path=save_path)
            if activate_preview_only:
                save_name = os.path.basename(datafolder[:-1]) + '_' +\
                    key + '_bin' +\
                    str(nb_frames_sorting_binning) +\
                    '_frame_selection_monitor_%s_%s' % (factor_minus0[idx_null],
                factor_plus0[idx_null])
                np.savez(save_path+save_name, Iminus=i_pm[0], Iplus = i_pm[1])
                continue

        '''
        Wavelength axis. One histogrm per value in this array will be created.
        The set of histogram will be fitted at once.
        '''
        wl_scale0 = data['wl_scale']

        '''
        Load the zeta coeff we need. if "wl_bounds" kew is sent, the return
        zeta coeff are the average over the bandwidth set by the tuple of this
        key.
        '''
        zeta_coeff = gff.get_zeta_coeff(zeta_coeff_path, wl_scale0, False)
        if not activate_zeta:
            for zkey in zeta_coeff.keys():
                if zkey != 'wl_scale':
                    zeta_coeff[zkey][:] = 1.


        '''
        Set zeta coeff linking null/antinull and photometric outputs into
        dedicated variables for clarity.
        '''
        zeta_minus_A = zeta_coeff['b%s%s' % (idx_photo[0]+1, key)]
        zeta_minus_B = zeta_coeff['b%s%s' % (idx_photo[1]+1, key)]
        zeta_plus_A = zeta_coeff['b%s%s' % (idx_photo[0]+1, key_antinull)]
        zeta_plus_B = zeta_coeff['b%s%s' % (idx_photo[1]+1, key_antinull)]

        # =====================================================================
        # Import dark data and get CDF of dark in interferometric outputs
        # =====================================================================
        '''
        Get histograms of dark current in the pair of
        photomoetry outputs.
        '''
        dark_IA, dark_IB = dark['photo'][0], dark['photo'][1]

        '''
        Get CDF of dark currents in interferometric outputs for generating
        random values in the MC function.
        '''
        if activate_remove_dark:
            dark_Iminus = np.zeros_like(dark['Iminus'])
            dark_Iplus = np.zeros_like(dark['Iplus'])
        else:
            dark_Iminus = dark['Iminus']
            dark_Iplus = dark['Iplus']
            
        std_dark_Iminus = np.std(dark_Iminus)
        dark_Iminus -= np.mean(dark_Iminus, 1, keepdims=True)
            
        dark_Iminus_axis, dark_Iminus_cdf = gff.get_dark_cdf(
            dark_Iminus, wl_scale0)
        dark_Iplus_axis, dark_Iplus_cdf = gff.get_dark_cdf(
            dark_Iplus, wl_scale0)
        
        # =====================================================================
        # Get the values of injection and spectrum
        # =====================================================================
        '''
        Get histograms of intensities in the pair of
        photomoetry outputs.

        Set photometries in dedicated variable into specific variables for
        clarity.
        A and B are the generic id of the beams for the processed baseline.
        '''
        data_IA, data_IB = data['photo'][0], data['photo'][1]

        '''
        Either get the CDF for rv generation or just keep the
        measurements if **activate_use_photometry** is ``True``.
        '''
        injection, spectra = gff.get_injection_and_spectrum(
            data_IA, data_IB, wl_scale0, (wl_min, wl_max))

        injection = np.array(injection)
        injectionbefore = injection.copy()
        injection_dark, spectra_dark = gff.get_injection_and_spectrum(
            dark_IA, dark_IB, wl_scale0)
        injection_dark = np.array(injection_dark)
        
        if activate_use_photometry:
            n_samp_per_loop = data_IA.shape[1]
            nloop = max((n_samp_total // n_samp_per_loop, 1))
            
        if activate_remove_dark:
            injection_dark[:] = 0.
            
            
        # =====================================================================
        # Get the distribution of the fringe blurring (piston_rms)
        # =====================================================================
        if activate_lbti_mode:
            sigma_eps = data['piston_rms']
            sigma_eps = np.radians(sigma_eps)
            sigma_eps *= 2200 / wl_scale0
            # sigma_eps = 2 * np.pi / wl_scale0 * sigma_eps
            sigma_eps = sigma_eps.reshape((1, -1))
            # sigma_eps = sigma_eps.mean(1, keepdims=True)
            sigma_eps_axis, sigma_eps_cdf = gff.get_dark_cdf(sigma_eps, wl_scale0)
            sigma_eps_axis = sigma_eps_axis[0]
            sigma_eps_cdf = sigma_eps_cdf[0]
        
        # =====================================================================
        # Compute the null
        # =====================================================================
        if activate_spectral_binning:
            Iminus, dummy = gff.binning(
                data['Iminus'], data['Iminus'].shape[0], 0, False)
            Iplus, dummy = gff.binning(
                data['Iplus'], data['Iplus'].shape[0], 0, False)
            wl_scale, dummy = gff.binning(wl_scale0, wl_scale0.size, 0, True)
        else:
            Iminus = data['Iminus']
            Iplus = data['Iplus']
            wl_scale = wl_scale0
            
        if activate_use_antinull:
            if key in nulls_to_invert:
                data_null = Iplus / Iminus
            else:
                data_null = Iminus / Iplus
        else:
            IA = data_IA.copy()
            IB = data_IB.copy()
            IA[IA <= 0] = 0
            IB[IB <= 0] = 0
            if key in nulls_to_invert:
                Iminus = IA*zeta_plus_A[:, None] + IB*zeta_plus_B[:, None] +\
                    2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A[:, None] *
                                                   zeta_plus_B[:, None])
                data_null = Iplus / Iminus
            else:
                Iplus = IA*zeta_minus_A[:, None] + IB*zeta_minus_B[:, None] +\
                    2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A[:, None] *
                                                   zeta_minus_B[:, None])
                data_null = Iminus / Iplus
                

        # =====================================================================
        # Filter null depth data set
        # =====================================================================
        print('Compute survival function and error bars')
        bin_bounds = bin_bounds0[idx_null]

        '''
        EXPERIMENTAL

        Removing frames for which the null depth is out of the histogram
        for at least one spectral channel.
        '''
        if activate_spectral_sorting:
            mask_null = np.array([(d >= bin_bounds[0]) & (
                d <= bin_bounds[1]) for d in data_null])
            mask_frames = np.unique(np.where(mask_null == False)[1])
            mask_null[:, mask_frames] = False
        else:
            mask_null = np.ones_like(data_null, dtype=bool)

        '''
        Creation of the x-axis of the histogram (one per wavelength)
        '''
        data_null = np.array([data_null[k][mask_null[k]]
                              for k in range(data_null.shape[0])])

        # =====================================================================
        # Basic estimation for classic calibration
        # =====================================================================
        if activate_save_classic_esti:
            classic_save = save_path + os.path.basename(datafolder[:-1]) + '_' +\
                key + '_' + '%03d' %(basin_hopping_nloop[0]) + '_classic_esti' +\
                    '_'+str(wl_min) +\
                '-' + str(wl_max) + '_files_' + str(nb_files_data[0]) +\
                '_' + str(nb_files_data[1])

            if activate_spectral_binning:
                classic_save = classic_save+'_sp'
                
            sys.stdout = Logger(classic_save + '_log.log')
            
            print('---------------------')
            print('Classic estimation of ' + key)
            print('%s-%02d-%02d %02dh%02d' % (datetime.now().year,
                                              datetime.now().month,
                                              datetime.now().day,
                                              datetime.now().hour,
                                              datetime.now().minute))
            print('Spectral bandwidth (in nm):%s,%s' % (wl_min, wl_max))
            print('Spectral binning %s' % activate_spectral_binning)
            print('Number of loaded points', data_null.shape[1], nb_files_data)
            print('Global binning of frames:', global_binning)
            print('Time to load files: %.3f s' %
                  (stop_loading - start_loading))
            print('activate phase sorting', activate_phase_sorting)
            print('activate_preview_only', activate_preview_only)
            print('factor_minus, factor_plus',
                  factor_minus0[idx_null], factor_plus0[idx_null])
            print('nb_frames_sorting_binning', nb_frames_sorting_binning)
           
            data_null2 = [d[d >= 0] for d in data_null]
            rms = [d.std() for d in data_null2]
            data_null3 = \
                [data_null2[k][(data_null2[k] >= data_null2[k].min()) &
                               (data_null2[k] <= data_null2[k].min()+rms[k])]
                 for k in range(len(data_null2))]
            avg = np.array([np.mean(elt) for elt in data_null3])
            std = np.array([np.std(elt) for elt in data_null3])

            classic_esti = np.array([avg, std, wl_scale])
            print('Null depth (avg, std, wl scale):')
            print(classic_esti[0])
            print(classic_esti[1])
            print(classic_esti[2])
            print('')
            print('')

            np.save(classic_save, classic_esti)
            sys.stdout.close()
            
            continue

        # =====================================================================
        # Compute histogram
        # =====================================================================
        '''
        Size of the sample of measured null depth.

        Last uses show that a fixed number of bins give good results.
        It is possible to uncomment the original version based on the
        sample size.
        '''
        sz = int(1e+6)  # size of the sample of measured null depth.
        sz_list = np.array([np.size(d[(d >= bin_bounds[0]) &
                                      (d <= bin_bounds[1])])
                            for d in data_null])
        sz = np.max(sz_list)  # size of the sample of measured null depth.

        null_axis = np.array([np.linspace(bin_bounds[0], bin_bounds[1], int(
            sz**0.5+1), retstep=False, dtype=np.float32)
            for i in range(data_null.shape[0])])

        '''
        Compute the histogram per spectral channel and add it in the list
        **null_pdf** and same for its error bars **null_pdf_err**.
        '''
        null_pdf = []
        null_pdf_err = []
        for wl in range(len(wl_scale)):
            pdf = np.histogram(data_null[wl], null_axis[wl], density=False)[0]
            pdf_size = np.sum(pdf)
            
            if normed:
                pdf = pdf / np.sum(pdf)
            else:
                pass
            null_pdf.append(pdf)
            pdf_err = gff.getErrorBinomNorm(pdf, pdf_size, normed=normed)
            null_pdf_err.append(pdf_err)

        null_pdf = np.array(null_pdf)
        null_pdf_err = np.array(null_pdf_err)
        
        # =====================================================================
        # Cropping photometries, Iminus and Iplus
        # to the kept values of null depths
        # =====================================================================
        if activate_spectral_binning:
            mask_null2 = np.tile(mask_null, (data_IA.shape[0], 1))
        else:
            mask_null2 = mask_null
        data_IA = np.array([data_IA[k][mask_null2[k]]
                            for k in range(data_IA.shape[0])])
        data_IB = np.array([data_IB[k][mask_null2[k]]
                            for k in range(data_IB.shape[0])])
        Iminus = np.array([Iminus[k][mask_null2[k]]
                           for k in range(Iminus.shape[0])])
        Iplus = np.array([Iplus[k][mask_null2[k]]
                          for k in range(Iplus.shape[0])])
        injection, dummy = gff.get_injection_and_spectrum(
            data_IA, data_IB, wl_scale0, (wl_min, wl_max))
        injection = np.array(injection)

        if activate_spectral_binning:
            data_IA = data_IA.sum(0).reshape((1, -1))
            data_IB = data_IB.sum(0).reshape((1, -1))
            
        # =====================================================================
        # Get the distribution of the photometries and selecting values
        # in the range of null depths
        # =====================================================================
        if activate_time_binning_photometry:
            if nb_frames_binning_photometry < 0:
                injection = injectionbefore.mean(axis=-1).reshape(2, -1)
            else:
                injection, dummy = gff.binning(
                    injection, nb_frames_binning_photometry, axis=1, avg=True)
                injection_dark, dummy = gff.binning(
                    injection_dark, nb_frames_binning_photometry, axis=1,
                    avg=True)

        if activate_dark_correction and not activate_use_photometry:
            injection_saved = injection.copy()

            if activate_photo_resampling:
                histo_injectionA = list(np.histogram(
                    injection[0], int(injection[0].size**0.5)+1, density=True))
                histo_injectionB = list(np.histogram(
                    injection[1], int(injection[1].size**0.5)+1, density=True))
                abscA = histo_injectionA[1][:-1] + \
                    np.diff(histo_injectionA[1])/2
                abscB = histo_injectionB[1][:-1] + \
                    np.diff(histo_injectionB[1])/2
                poptA, pcovA = curve_fit(gaussian, abscA, histo_injectionA[0],
                                         p0=[np.max(histo_injectionA[0]),
                                             abscA[np.argmax(
                                                 histo_injectionA[0])],
                                             100])
                poptB, pcovB = curve_fit(gaussian, abscB, histo_injectionB[0],
                                         p0=[np.max(histo_injectionB[0]),
                                             abscB[np.argmax(
                                                 histo_injectionB[0])],
                                             100])
                injection_saved = injection.copy()
                injection_mean = np.array([poptA[1], poptB[1]])
                injection_var = np.array([poptA[2]**2, poptB[2]**2])

                injection_dk_var = injection_dark.var(axis=1)
                injection_corrected_std = (
                    injection_var - injection_dk_var)**0.5
                injection = np.array([np.random.normal(injection_mean[k],
                                                       injection_corrected_std[k],
                                                       injection.shape[1])
                                      for k in range(injection.shape[0])])
            else:
                mean_data, var_data = np.mean(
                    injection, axis=-1), np.var(injection, axis=-1)
                mean_dark, var_dark = np.mean(
                    injection_dark, axis=-1), np.var(injection_dark, axis=-1)

                injection = (injection - mean_data[:, None]) * \
                    ((var_data[:, None]-var_dark[:, None])/var_data[:, None]
                     )**0.5 + mean_data[:, None] - mean_dark[:, None]

                if np.any(np.isnan(injection)) or np.any(np.isinf(injection)):
                    print('Restore injection')
                    injection = injection_saved.copy()

        data_IA_axis = cp.linspace(injection[0].min(), injection[0].max(),
                                   np.size(np.unique(injection[0])),
                                   dtype=cp.float32)
        cdf_data_IA = gff.computeCdf(data_IA_axis, injection[0], 'cdf', True)
        cdf_data_IA = cp.array(cdf_data_IA, dtype=cp.float32)

        data_IB_axis = cp.linspace(injection[1].min(), injection[1].max(),
                                   np.size(np.unique(injection[1])),
                                   dtype=cp.float32)
        cdf_data_IB = gff.computeCdf(data_IB_axis, injection[1], 'cdf', True)
        cdf_data_IB = cp.array(cdf_data_IB, dtype=cp.float32)

        # =====================================================================
        # Model fitting and Chi2 map
        # =====================================================================
        ''' Select the bounds of the baseline (null) to process '''
        bounds_mu = bounds_mu0[idx_null]
        bounds_sig = bounds_sig0[idx_null]
        bounds_na = bounds_na0[idx_null]
        # Compile them into a readable tuple called by the TRF algorithm
        bounds_fit = np.array(([bounds_na[0], bounds_mu[0], bounds_sig[0]],
                      [bounds_na[1], bounds_mu[1], bounds_sig[1]]))
        if not activate_use_antinull and not activate_lbti_mode:
            number_of_Ir = 1  # wl_scale0.size
            bounds_fit = np.array(([bounds_na[0], bounds_mu[0], bounds_sig[0]] +
                          [0] * number_of_Ir+[1e-6]*number_of_Ir,
                          [bounds_na[1], bounds_mu[1], bounds_sig[1]] +
                          [10] * number_of_Ir+[10]*number_of_Ir))
            diffstep = diffstep + [0.1]*number_of_Ir+[0.1]*number_of_Ir
            xscale = np.append(xscale, [[1]*number_of_Ir, [1]*number_of_Ir])

        wl_scale_saved = wl_scale.copy()
        
        rvu_opd = None
        rvu_sigmaeps = None
        rvu_injectionA = None
        rvu_injectionB = None
        rvu_darkm = [None]*wl_scale0.size
        rvu_darkp = [None]*wl_scale0.size
        if activate_rvu == True:
            rvu_opd = cp.random.uniform(0, 1, size=n_samp_per_loop, dtype=cp.float32)
            rvu_sigmaeps = cp.random.uniform(0, 1, size=n_samp_per_loop, dtype=cp.float32)
            rvu_injectionA = cp.random.uniform(0, 1, size=n_samp_per_loop, dtype=cp.float32)
            rvu_injectionB = cp.random.uniform(0, 1, size=n_samp_per_loop, dtype=cp.float32)
            rvu_darkm = cp.random.uniform(0, 1, size=(wl_scale0.size, n_samp_per_loop), dtype=cp.float32)
            rvu_darkp = cp.random.uniform(0, 1, size=(wl_scale0.size, n_samp_per_loop), dtype=cp.float32)
        
        for idx_basin, basin_hopping_count in enumerate(
                range(basin_hopping_nloop[0], basin_hopping_nloop[1])):
            if not skip_fit and not chi2_map_switch:
                plt.close('all')

            if not chi2_map_switch:
                # Save the content written in the console into a txt file
                sys.stdout = Logger(
                    save_path+'%s_%03d_basin_hop.log' % (key,
                                                         basin_hopping_count))

            # Save the reduced Chi2 of the different basin hop
            chi2_liste = []
            # Save the optimal parameters of the different basin hop
            popt_liste = []
            # Save the errors on fitted parameters of the different basin hop
            uncertainties_liste = []
            # Save the initial guesses of the different basin hop
            init_liste = []
            # Save the covariance matrix of the different basin hop
            pcov_liste = []
            # Save the termination condition of the different basin hop
            termination_liste = []

            print('-------------')
            print(basin_hopping_count)
            print('-------------')
            print('Execution mode: ' + msg)
            print('Baseline '+key)
            print('%s-%02d-%02d %02dh%02d' % (datetime.now().year,
                                              datetime.now().month,
                                              datetime.now().day,
                                              datetime.now().hour,
                                              datetime.now().minute))
            print('Spectral bandwidth (in nm):%s,%s' % (wl_min, wl_max))
            print('Spectral binning %s' % activate_spectral_binning)
            print('Time binning of photometry %s (%s)' % (
                activate_time_binning_photometry,
                nb_frames_binning_photometry))
            print('Bin bounds and number of bins %s %s' %
                  (bin_bounds, int(sz**0.5)))
            print('Boundaries (na, mu, sig):')
            print(str(bounds_na)+str(bounds_mu)+str(bounds_sig))
            print('Number of elements ', n_samp_total)
            print('Number of loaded points', data_null.shape[1], nb_files_data)
            print('Number of loaded dark points',
                  dark['Iminus'].shape[1], nb_files_dark)
            print('Global binning of frames:', global_binning)
            print('Zeta file', os.path.basename(config['zeta_coeff_path']))
            print('Time to load files: %.3f s' %
                  (stop_loading - start_loading))
            print('activate phase sorting', activate_phase_sorting)
            print('activate_preview_only', activate_preview_only)
            print('factor_minus, factor_plus',
                  factor_minus0[idx_null], factor_plus0[idx_null])
            print('nb frames sorting binning', nb_frames_sorting_binning)
            print('Injection - shape array', injection.shape)
            print('Injection dark - shape array', injection_dark.shape)
            print('Remove dark', activate_remove_dark)
            print('Type of optimizer', select_optimizer)
            print('Normed PDF', normed)
            print('activate_rvu', activate_rvu)
            print('activate_mcmc', activate_mcmc)
            print('')

            '''
            Model fitting initial guess
            '''
            # Create the set of initial guess for each hop
            if idx_basin > 0 and activate_random_init_guesses:
                mu_opd, sig_opd, na = basin_hoppin_values(
                    mu_opd0[idx_null], sig_opd0[idx_null], na0[idx_null],
                    bounds_mu, bounds_sig, bounds_na)
            else:
                mu_opd = mu_opd0[idx_null]
                sig_opd = sig_opd0[idx_null]
                na = na0[idx_null]

            '''
            Model fitting
            '''
            if not chi2_map_switch and not activate_mcmc:
                guess_na = na
                initial_guess = [guess_na, mu_opd, sig_opd]
                if not activate_use_antinull and not activate_lbti_mode:
                    initial_guess = [guess_na, mu_opd, sig_opd] + \
                        [1]*number_of_Ir + [1]*number_of_Ir
                initial_guess = np.array(initial_guess, dtype=np.float64)

                if skip_fit:
                    '''
                    No fit is perforend here, just load the values
                    in the initial guess and compute the histogram.
                    '''
                    print('Direct display')
                    count = 0.
                    start = time()
                    out = nsc_function(null_axis, *initial_guess, normed=normed)
                    stop = time()
                    print('Duration:', stop-start)
                    na_opt = na
                    uncertainties = np.zeros(3)
                    popt = (np.array([na, mu_opd, sig_opd]), np.ones((3, 3)))
                    
                    if select_optimizer == 0:
                        chi2 = 1/(null_pdf.size-popt[0].size) * np.sum(
                            (null_pdf.ravel() - out)**2/null_pdf_err.ravel()**2)
                    elif select_optimizer == 1:
                        out = out.reshape((wl_scale.size, -1))
                        out = out / out.sum(1)[:,None] * null_pdf.sum(1)[:, None]
                        out = out.ravel()
                        chi2 = gff.likelihood(popt[0], null_pdf, nsc_function, null_axis, n_samp_per_loop, use_this_model=out)
                    elif select_optimizer == 2:
                        out = out.reshape((wl_scale.size, -1))
                        out = out / out.sum(1)[:,None] * null_pdf.sum(1)[:, None]
                        out = out.ravel()
                        chi2 = gff.likelihoodChi2(popt[0], null_pdf, nsc_function, null_axis, null_pdf_err, n_samp_per_loop, use_this_model=out)
                    term_status = None
                    print(label_optimizer, chi2)

                else:
                    '''
                    Fit is done here
                    '''
                    print('Model fitting')
                    count = 0.
                    init_liste.append(initial_guess)

                    start = time()
                    if select_optimizer == 0:
                        popt = gff.curvefit(nsc_function, null_axis,
                                            null_pdf.ravel(),
                                            p0=initial_guess,
                                            sigma=null_pdf_err.ravel(),
                                            bounds=bounds_fit, diff_step=diffstep,
                                            x_scale=xscale)
                    elif select_optimizer == 1:
                        popt = gff.optimize(nsc_function, gff.likelihood, null_axis, null_pdf,
                                            p0=initial_guess,
                                            bounds=bounds_fit, diff_step=diffstep)
                    elif select_optimizer == 2:
                        popt = gff.optimize(nsc_function, gff.likelihoodChi2, null_axis, null_pdf,
                                            yerr=null_pdf_err, p0=initial_guess,
                                            bounds=bounds_fit, diff_step=diffstep)
                    # Other outpus of the **curvefit** function
                    res = popt[2]
                    # Optimal parameters found
                    popt = popt[:2]
                    print('Termination:', res.message)  # Termination condition
                    # Value associated by the termination condition
                    term_status = res.status

                    stop = time()
                    print('Termination', term_status)
                    print('Duration:', stop - start)

                    # Histogram computed according to the optimal parameters
                    out = nsc_function(null_axis, *popt[0], normed=normed)
                    # Errors on the optimal parameters
                    uncertainties = np.diag(popt[1])**0.5
                    # Reduced Chi2
                    if select_optimizer == 0:
                        chi2 = 1/(null_pdf.size-popt[0].size) *\
                            np.sum((null_pdf.ravel() - out) ** 2 /
                                   null_pdf_err.ravel()**2)
                    elif select_optimizer == 1 or select_optimizer == 2:
                        chi2 = res.fun
                        out = out.reshape((wl_scale.size, -1))
                        out = out / out.sum(1)[:,None] * null_pdf.sum(1)[:, None]
                        out = out.ravel()

                    print(label_optimizer, chi2)

                    '''
                    Display in an easy-to-read way this key information
                    (optimal parameters, error and reduced Chi2)
                    '''
                    na_opt = popt[0][0]
                    print('******')
                    print(popt[0])
                    print(popt[1])
                    if select_optimizer == 0 or select_optimizer == 2:
                        print(uncertainties*chi2**0.5)
                    else:
                        print(uncertainties)
                    print(chi2)
                    print('******')

                    '''
                    Save input and the outputs of the fit into a npz file.
                    One per basin hop.
                    '''
                    np.savez(save_path+'%s_%03d_' % (key, basin_hopping_count) +
                             os.path.basename(file_path[:-1]),
                             chi2=chi2, popt=[na_opt] +
                             [elt for elt in popt[0][1:]],
                             uncertainties=uncertainties,
                             init=[guess_na]+list(initial_guess[1:]),
                             termination=np.array([term_status]),
                             nsamp=np.array([n_samp_per_loop]), wl=wl_scale)

                chi2_liste.append(chi2)
                popt_liste.append([na_opt]+[elt for elt in popt[0][1:]])
                uncertainties_liste.append(uncertainties)
                termination_liste.append(term_status)
                pcov_liste.append(popt[1])

                '''
                Display the results of the fit for publishing purpose
                '''
                nb_rows_plot = (wl_scale.size//2) + (wl_scale.size % 2)
                wl_idx0 = np.arange(wl_scale.size)[::-1][:-1]
                # Subset of wavelength displayed in one figure
                wl_idx0 = list(gff.divide_chunks(wl_idx0, nb_rows_plot*2))
                labelsz = 28
                subplot_letters = list('abcdefghijklmnopqrstuvwxyz')
                for wl_idx in wl_idx0:
                    f = plt.figure(figsize=(20, 3.60*nb_rows_plot))
                    count = 0
                    axs = []
                    for wl in wl_idx:
                        if len(wl_idx) > 1:
                            ax = f.add_subplot(nb_rows_plot, 2, count+1)
                        else:
                            ax = f.add_subplot(1, 1, count+1)
                        axs.append(ax)
                        ax.ticklabel_format(axis='y', style='sci',
                                            useMathText=True, scilimits=(0, 0))
                        ax.errorbar(null_axis[wl][:-1], null_pdf[wl],
                                    yerr=null_pdf_err[wl], fmt='.',
                                    markersize=20, label='Data')
                        if activate_draw_model:
                            ax.errorbar(null_axis[wl][:-1],
                                        out.reshape((wl_scale.size, -1))[wl],
                                        markersize=5, lw=10, alpha=0.8,
                                        label='Fit')
                        ax.grid()
                        if list(wl_idx).index(wl) >= len(wl_idx)-2 or\
                                len(wl_idx) == 1:
                            ax.set_xlabel('Null depth', size=labelsz)
                        if count % 2 == 0:
                            ax.set_ylabel('Frequency', size=labelsz)
                            plt.savefig('deleteme.png')
                            exponent_text = ax.yaxis.get_offset_text().get_text()
                            label = ax.yaxis.get_label().get_text()
                            ax.yaxis.offsetText.set_visible(False)
                            ax.yaxis.set_label_text(
                                gff.update_label(label, exponent_text))
                        else:
                            ax.yaxis.offsetText.set_visible(False)
                        ax.set_xticks(ax.get_xticks()[::2])  # , size=30)
                        ax.tick_params(axis='both', labelsize=labelsz-2)
                        exponent = np.floor(np.log10(null_pdf.max()))
                        ax.set_ylim(-10**(exponent)/10, max(null_pdf.max(), out.max())*1.05)
                        ax.text(0.7, 0.8, '%.0f nm' % (
                            wl_scale[wl]), va='center', transform=ax.transAxes,
                            fontsize=labelsz)
                        
                        if count % 2 == 0:
                            ax.text(-0.15, 1.1, subplot_letters[count]+')',
                                    va='center', transform=ax.transAxes,
                                    fontsize=labelsz, fontweight='bold')
                        else:
                            ax.text(-0.05, 1.1, subplot_letters[count]+')',
                                    va='center', transform=ax.transAxes,
                                    fontsize=labelsz, fontweight='bold')
                        count += 1

                    plt.tight_layout(rect=[0.02, 0, 1, 1])
                    string = key + '_' + '%03d' % (basin_hopping_count) +\
                        '_' +\
                        str(wl_min) + '-' + str(wl_max) + '_' +\
                        os.path.basename(datafolder[:-1]) +\
                        '_%s' % int(np.around(wl_scale[wl_idx[-1]]))
                    if not activate_oversampling:
                        string = string + '_nooversamplinginmodel'
                    if not activate_zeta:
                        string = string + '_nozetainmodel'
                    if not skip_fit:
                        string = string + '_fit_pdf'
                    if activate_spectral_binning:
                        string = string + '_sb'
                    plt.savefig(save_path+string+'_compiled.png', dpi=150)
                    plt.savefig(save_path+string+'_compiled.pdf', format='pdf')
                    plt.close()

                '''
                Display for archive and monitoring purpose
                '''
                nb_rows_plot = 3
                wl_idx0 = np.arange(wl_scale.size)[::-1]
                # Subset of wavelength displayed in one figure
                wl_idx0 = list(gff.divide_chunks(wl_idx0, nb_rows_plot*2))

                for wl_idx in wl_idx0:
                    f = plt.figure(figsize=(19.20, 3.60*nb_rows_plot))
                    text_params = '%s ' % key+'Fitted values: ' +\
                        'Na2$ = %.2E \pm %.2E$, ' % (na_opt,
                                                    uncertainties[0]) +\
                        r'$\mu_{OPD} = %.2E \pm %.2E$ nm, ' % (popt[0][1],
                                                               uncertainties[1]) +\
                        '\n' + r'$\sigma_{OPD} = %.2E \pm %.2E$ nm,' % (
                            popt[0][2], uncertainties[2]) +\
                        ' '+label_optimizer+' = %.2E ' % (chi2) +\
                        '(Duration = %.3f s)' % (stop-start)
                    count = 0
                    axs = []
                    for wl in wl_idx:
                        if len(wl_idx) > 1:
                            ax = f.add_subplot(nb_rows_plot, 2, count+1)
                        else:
                            ax = f.add_subplot(1, 1, count+1)
                        axs.append(ax)
                        plt.title('%.0f nm' % wl_scale[wl], size=30)
                        plt.errorbar(
                            null_axis[wl][:-1], null_pdf[wl],
                            yerr=null_pdf_err[wl],
                            fmt='.', markersize=10, label='Data')
                        if activate_draw_model:
                            plt.errorbar(null_axis[wl][:-1],
                                         out.reshape((wl_scale.size, -1))[wl],
                                         markersize=10, lw=3, alpha=0.8,
                                         label='Fit')
                        plt.grid()
    #                    plt.legend(loc='best', fontsize=25)
                        if list(wl_idx).index(wl) >= len(wl_idx)-2 or\
                                len(wl_idx) == 1:
                            plt.xlabel('Null depth', size=30)
                        if count % 2 == 0:
                            plt.ylabel('Frequency', size=30)
                        plt.xticks(size=22)
                        plt.yticks(size=22)
                        exponent = np.floor(np.log10(null_pdf.max()))
                        plt.ylim(-10**(exponent)/10, null_pdf.max()*1.05)
                        # plt.xlim(-0.01, 0.5)
                        count += 1
                    plt.tight_layout(rect=[0., 0., 1, 0.88])
    #                plt.tight_layout()
                    if len(wl_idx) > 1:
                        axs[0].text(0.3, 1.52, text_params, va='center',
                                    transform=axs[0].transAxes,
                                    bbox=dict(boxstyle="square",
                                              facecolor='white'), fontsize=20)
                    else:
                        axs[0].text(0.25, 1.15, text_params, va='center',
                                    transform=axs[0].transAxes,
                                    bbox=dict(boxstyle="square",
                                              facecolor='white'), fontsize=17)
                    string = key + '_' + '%03d' % (basin_hopping_count) + \
                        '_' +\
                        str(wl_min) + '-' + str(wl_max) + '_' +\
                        os.path.basename(datafolder[:-1]) +\
                        '_%s' % int(np.around(wl_scale[wl_idx[-1]]))
                    if not activate_oversampling:
                        string = string + '_nooversamplinginmodel'
                    if not activate_zeta:
                        string = string + '_nozetainmodel'
                    if not skip_fit:
                        string = string + '_fit_pdf'
                    if activate_spectral_binning:
                        string = string + '_sb'
                    plt.savefig(save_path+string+'.png', dpi=150)

                '''
                Display the details of the fit: make sure the reconstructed
                null and antinull match with the real ones
                '''
                liste_rv_interfminus = cp.asnumpy(liste_rv_interfminus)
                liste_rv_interfplus = cp.asnumpy(liste_rv_interfplus)

                if activate_spectral_binning:
                    liste_rv_interfminus = np.nansum(
                        liste_rv_interfminus, 0).reshape((1, -1))
                    liste_rv_interfplus = np.nansum(
                        liste_rv_interfplus, 0).reshape((1, -1))
                else:
                    liste_rv_interfminus = [
                        elt[~np.isnan(elt)] for elt in liste_rv_interfminus]
                    liste_rv_interfplus = [
                        elt[~np.isnan(elt)] for elt in liste_rv_interfplus]

                histom = [np.histogram(Iminus[k], bins=int(
                    Iminus[k].size**0.5), density=True)
                    for k in range(wl_scale.size)]
                histom2 = [np.histogram(liste_rv_interfminus[k], bins=int(
                    liste_rv_interfminus[k].size**0.5), density=True)
                    for k in range(wl_scale.size)]
                histop = [np.histogram(Iplus[k], bins=int(
                    Iplus[k].size**0.5), density=True)
                    for k in range(wl_scale.size)]
                histop2 = [np.histogram(liste_rv_interfplus[k], bins=int(
                    liste_rv_interfplus[k].size**0.5), density=True)
                    for k in range(wl_scale.size)]
                histodkm = [np.histogram(dark['Iminus'][k], bins=int(
                    dark['Iminus'][k].size**0.5), density=True)
                    for k in range(wl_scale.size)]
                histodkp = [np.histogram(dark['Iplus'][k], bins=int(
                    dark['Iplus'][k].size**0.5), density=True)
                    for k in range(wl_scale.size)]

                for wl_idx in wl_idx0:
                    f = plt.figure(figsize=(19.20, 10.80))
                    count = 0
                    axs = []
                    for wl in wl_idx:
                        if len(wl_idx) > 1:
                            ax = f.add_subplot(nb_rows_plot, 2, count+1)
                        else:
                            ax = f.add_subplot(1, 1, count+1)
                        axs.append(ax)
                        plt.title('%.0f nm' % wl_scale[wl], size=20)
                        plt.plot(histom[wl][1][:-1], histom[wl]
                                 [0], '.', label='Iminus')
                        plt.plot(histop[wl][1][:-1], histop[wl]
                                 [0], '.', label='Iplus')
                        if activate_draw_model:
                            plt.plot(histom2[wl][1][:-1],
                                     histom2[wl][0], label='rv minus')
                            plt.plot(histop2[wl][1][:-1],
                                     histop2[wl][0], label='rv plus')
                        plt.plot(histodkm[wl][1][:-1],
                                 histodkm[wl][0], label='Dark minus')
                        plt.plot(histodkp[wl][1][:-1],
                                 histodkp[wl][0], label='Dark plus')
                        plt.grid()
                        plt.xticks(size=15)
                        plt.yticks(size=15)
                        maxi = max([max([max(elt[0]) for elt in histom]),
                                    max([max(elt[0]) for elt in histom2]),
                                    max([max(elt[0]) for elt in histop]),
                                    max([max(elt[0]) for elt in histop2]),
                                    max([max(elt[0]) for elt in histodkm]),
                                    max([max(elt[0]) for elt in histodkp])])
                        exponent = np.floor(np.log10(maxi))
                        plt.ylim(-10**(exponent)/10, maxi*1.05)
                        if count % 2 == 0:
                            plt.ylabel('Frequency', size=20)
                        if count == 0:
                            plt.legend(loc='best', ncol=3, fontsize=15)
                        if list(wl_idx).index(wl) <= 1 or len(wl_idx) == 1:
                            plt.xlabel('Flux (AU)', size=20)
                        count += 1
                    plt.tight_layout()
                    string = key + '_%03d' % (basin_hopping_count) +\
                        '_details_' + str(wl_min) + '-' + str(wl_max) + '_' +\
                        os.path.basename(datafolder[:-1]) +\
                        '_%s' % int(np.around(wl_scale[wl_idx[-1]]))
                    if not activate_oversampling:
                        string = string + '_nooversamplinginmodel'
                    if not activate_zeta:
                        string = string + '_nozetainmodel'
                    if not skip_fit:
                        string = string + '_fit_pdf'
                    if activate_spectral_binning:
                        string = string + '_sb'
                    plt.savefig(save_path+string+'.png', dpi=150)

                '''
                Plot the histogram of the photometries
                '''
                liste_rv_IA = cp.asnumpy(liste_rv_IA)
                liste_rv_IB = cp.asnumpy(liste_rv_IB)
               
                # gff.plot_photometries_histo(data_IA, dark['photo'][0], wl_scale,
                #                             wl_idx0, nb_rows_plot, count,
                #                             null_table[key][1][0]+1, save_path,
                #                             activate_spectral_binning, skip_fit,
                #                             key, basin_hopping_count, wl_min,
                #                             wl_max, datafolder)
                # gff.plot_photometries_histo(data_IB, dark['photo'][1], wl_scale,
                #                             wl_idx0, nb_rows_plot, count,
                #                             null_table[key][1][1]+1, save_path,
                #                             activate_spectral_binning, skip_fit,
                #                             key, basin_hopping_count, wl_min,
                #                             wl_max, datafolder)
                gff.plot_photometries_histo(data_IA, liste_rv_IA, wl_scale,
                                            wl_idx0, nb_rows_plot, count,
                                            null_table[key][1][0]+1, save_path,
                                            activate_spectral_binning, skip_fit,
                                            key, basin_hopping_count, wl_min,
                                            wl_max, datafolder)
                gff.plot_photometries_histo(data_IB, liste_rv_IB, wl_scale,
                                            wl_idx0, nb_rows_plot, count,
                                            null_table[key][1][1]+1, save_path,
                                            activate_spectral_binning, skip_fit,
                                            key, basin_hopping_count, wl_min,
                                            wl_max, datafolder)
                '''
                Injection histogram
                '''
                if not activate_use_photometry:
                    rv_injectionA = cp.asnumpy(rv_injectionA)
                    rv_injectionB  = cp.asnumpy(rv_injectionB)
                    f = plt.figure(figsize=(19.20, 10.80))
                    histo_injectionA = np.histogram(
                        injection[0], int(injection[0].size**0.5), density=True)
                    histo_injectionB = np.histogram(
                        injection[1], int(injection[1].size**0.5), density=True)
                    histo_injection_dkA = np.histogram(injection_dark[0], int(
                        injection_dark[0].size**0.5), density=True)
                    histo_injection_dkB = np.histogram(injection_dark[1], int(
                        injection_dark[1].size**0.5), density=True)                
                    plt.title('%.0f nm' % wl_scale[wl], size=20)
                    plt.plot(histo_injectionA[1][:-1], histo_injectionA[0], 'o',
                             markersize=5, lw=3,
                             label='Injection %s' % (null_table[key][1][0]+1))
                    plt.plot(histo_injectionB[1][:-1], histo_injectionB[0], 'o',
                             markersize=5, lw=3,
                             label='Injection %s' % (null_table[key][1][1]+1))
                    if not activate_remove_dark:
                        plt.plot(histo_injection_dkA[1][:-1], histo_injection_dkA[0],
                                  '+', markersize=8,
                                  label='Injection dark %s' % (null_table[key][1][0]+1))
                        plt.plot(histo_injection_dkB[1][:-1], histo_injection_dkB[0],
                                  '+', markersize=8,
                                  label='Injection dark %s' % (null_table[key][1][1]+1))
                    if activate_dark_correction:
                        histo_injectionA_before = np.histogram(
                            injection_saved[0], int(injection_saved[0].size**0.5),
                            density=True)
                        histo_injectionB_before = np.histogram(
                            injection_saved[1], int(injection_saved[1].size**0.5),
                            density=True)
                        plt.plot(histo_injectionA_before[1][:-1],
                                 histo_injectionA_before[0], 'x',
                                 markersize=8,
                                 label='Injection %s before correction' % (
                                     null_table[key][1][0]+1))
                        plt.plot(histo_injectionB_before[1][:-1],
                                 histo_injectionB_before[0], 'x',
                                 markersize=8,
                                 label='Injection %s before correction' % (
                                     null_table[key][1][1]+1))
                    plt.grid()
                    plt.legend(loc='best', fontsize=15)
                    plt.xlabel('Flux (AU)', size=20)
                    plt.ylabel('Frequency', size=20)
                    plt.xticks(size=15)
                    plt.yticks(size=15)
                    plt.tight_layout()
                    string = key + '_' + '%03d' % (basin_hopping_count) +\
                        '_injection_' + str(wl_min) + '-' + str(wl_max) + '_' +\
                        os.path.basename(datafolder[:-1]) +\
                        '_%.0f' % (wl_scale[wl_idx[-1]])
                    if not skip_fit:
                        string = string + '_fit_pdf'
                    if activate_spectral_binning:
                        string = string + '_sb'
                    plt.savefig(save_path+string+'.png', dpi=150)
                
                
                # '''
                # Sigma Eps Histogram
                # '''
                # sigma_eps_histo = np.histogram(sigma_eps, int(sigma_eps.size**0.5), density=True)
                # rv_sigma_histo = np.histogram(cp.asnumpy(rv_sigma_eps), int(cp.asnumpy(rv_sigma_eps).size**0.5), density=True)
                # f = plt.figure(figsize=(19.20, 10.80))
                # plt.plot(sigma_eps_histo[1][:-1], sigma_eps_histo[0], 'o', markersize=5, label='sigma eps')
                # plt.plot(rv_sigma_histo[1][:-1], rv_sigma_histo[0], '+', markersize=8, lw=3, label='rv')
                # plt.grid()
                # plt.legend(loc='best', fontsize=15)
                # plt.xlabel('Phase (rad)', size=20)
                # plt.ylabel('Frequency', size=20)
                # plt.xticks(size=15)
                # plt.yticks(size=15)
                # plt.tight_layout()                
                # string = key + '_' + '%03d' % (basin_hopping_count) +\
                #     '_sigma_eps_' + str(wl_min) + '-' + str(wl_max) + '_' +\
                #     os.path.basename(datafolder[:-1]) +\
                #     '_%.0f' % (wl_scale[wl_idx[-1]])
                # if not skip_fit:
                #     string = string + '_fit_pdf'
                # if activate_spectral_binning:
                #     string = string + '_sb'
                # plt.savefig(save_path+string+'.png', dpi=150)
                
                # '''
                # Plot the darks and RV darks
                # '''
                # rv_dark_Iminus, rv_dark_Iplus
                
                # rv_darkm_histo = [np.histogram(cp.asnumpy(liste_rv_dark_Iminus[k]),
                #                                bins=int(
                #                             liste_rv_dark_Iminus[k].size**0.5),
                #                             density=True) 
                #                   for k in range(wl_scale.size)]
                # rv_darkp_histo = [np.histogram(cp.asnumpy(liste_rv_dark_Iplus[k]),
                #                                bins=int(
                #                             liste_rv_dark_Iplus[k].size**0.5),
                #                             density=True) 
                #                   for k in range(wl_scale.size)]
                
                # for wl_idx in wl_idx0:
                #     f = plt.figure(figsize=(19.20, 10.80))
                #     count = 0
                #     axs = []
                #     for wl in wl_idx:
                #         if len(wl_idx) > 1:
                #             ax = f.add_subplot(nb_rows_plot, 2, count+1)
                #         else:
                #             ax = f.add_subplot(1, 1, count+1)
                #         axs.append(ax)
                #         plt.title('%.0f nm' % wl_scale[wl], size=20)
                #         plt.plot(histodkm[wl][1][:-1],
                #                  histodkm[wl][0], label='Dark minus', lw=3)
                #         # plt.plot(histodkp[wl][1][:-1],
                #         #          histodkp[wl][0], label='Dark plus')
                #         plt.plot(rv_darkm_histo[wl][1][:-1],
                #                  rv_darkm_histo[wl][0], label='RV Dark minus')        
                #         # plt.plot(rv_darkp_histo[wl][1][:-1],
                #         #          rv_darkp_histo[wl][0], label='RV Dark plus')                              
                #         plt.grid()
                #         plt.xticks(size=15)
                #         plt.yticks(size=15)
                #         # maxi = max([max([max(elt[0]) for elt in rv_darkm_histo]),
                #         #             max([max(elt[0]) for elt in rv_darkp_histo]),
                #         #             max([max(elt[0]) for elt in histodkm]),
                #         #             max([max(elt[0]) for elt in histodkp])])
                #         maxi = max([max([max(elt[0]) for elt in rv_darkm_histo]),
                #                     max([max(elt[0]) for elt in histodkm])])                        
                #         exponent = np.floor(np.log10(maxi))
                #         plt.ylim(-10**(exponent)/10, maxi*1.05)
                #         if count % 2 == 0:
                #             plt.ylabel('Frequency', size=20)
                #         if count == 0:
                #             plt.legend(loc='best', ncol=3, fontsize=15)
                #         if list(wl_idx).index(wl) <= 1 or len(wl_idx) == 1:
                #             plt.xlabel('Flux (AU)', size=20)
                #         count += 1
                #     plt.tight_layout()
                #     string = key + '_%03d' % (basin_hopping_count) +\
                #         '_darks_' + str(wl_min) + '-' + str(wl_max) + '_' +\
                #         os.path.basename(datafolder[:-1]) +\
                #         '_%s' % int(np.around(wl_scale[wl_idx[-1]]))
                #     if not activate_oversampling:
                #         string = string + '_nooversamplinginmodel'
                #     if not activate_zeta:
                #         string = string + '_nozetainmodel'
                #     if not skip_fit:
                #         string = string + '_fit_pdf'
                #     if activate_spectral_binning:
                #         string = string + '_sb'
                #     plt.savefig(save_path+string+'.png', dpi=150)
            elif chi2_map_switch:
                ''' Map the parameters space '''
                print('Mapping parameters space')
                count = 0
                map_na, step_na = np.linspace(
                    bounds_na[0], bounds_na[1], map_na_sz, endpoint=False,
                    retstep=True)
                map_mu_opd, step_mu = np.linspace(
                    bounds_mu[0], bounds_mu[1], map_mu_sz, endpoint=False,
                    retstep=True)
                map_sig_opd, step_sig = np.linspace(
                    bounds_sig[0], bounds_sig[1], map_sig_sz, endpoint=False,
                    retstep=True)
                chi2map = []
                start = time()
                for visi in map_na:
                    temp1 = []
                    for o in map_mu_opd:
                        temp2 = []
                        for s in map_sig_opd:
                            parameters = np.array([visi, o, s])
                            out = nsc_function(null_axis, *parameters, normed=normed)
                            
                            if select_optimizer == 0:
                                tmp = (null_pdf.ravel() - out)**2 / null_pdf_err.ravel()**2
                                value = 1/(null_pdf.size-parameters.size) * \
                                    np.sum(tmp)
                            elif select_optimizer == 1:
                                value = gff.likelihood(parameters, null_pdf, nsc_function, null_axis, n_samp_per_loop, use_this_model=out)
                            elif select_optimizer == 2:
                                out = out.reshape((wl_scale.size, -1))
                                out = out / out.sum(1)[:,None] * null_pdf.sum(1)[:, None]
                                out = out.ravel()                                
                                value = gff.likelihoodChi2(parameters, null_pdf, nsc_function, null_axis, n_samp_per_loop, use_this_model=out, data_err=null_pdf_err)

                            temp2.append([value, visi, o, s])
                        temp1.append(temp2)
                    chi2map.append(temp1)
                stop = time()
                chi2map = np.array(chi2map)
                print('Duration: %.3f s' % (stop-start))

                if activate_spectral_binning:
                    chi2_savepath = save_path + \
                        label_optimizer+'map_%s_%03d_%.0f-%.0f_sp' % (
                            key, basin_hopping_count, wl_min, wl_max)
                else:
                    chi2_savepath = save_path + \
                        label_optimizer+'map_%s_%03d_%.0f-%.0f' % (key,
                                                       basin_hopping_count,
                                                       wl_min,
                                                       wl_max)
                np.savez(chi2_savepath, value=chi2map, na=map_na,
                         mu=map_mu_opd, sig=map_sig_opd, wl=wl_scale)

                chi2map2 = chi2map[:, :, :, 0]
                chi2map2[np.isnan(chi2map2)] = np.nanmax(chi2map[:, :, :, 0])
                argmin = np.unravel_index(np.argmin(chi2map2), chi2map2.shape)
                print('Min in param space', chi2map2.min(), map_na[argmin[0]],
                      map_mu_opd[argmin[1]], map_sig_opd[argmin[2]])
                print('Indexes are:', argmin)
                fake = chi2map2.copy()
                fake[argmin] = chi2map2.max()
                argmin2 = np.unravel_index(np.argmin(fake), chi2map2.shape)
                print('2nd min in param space',
                      chi2map2[argmin2], map_na[argmin2[0]],
                      map_mu_opd[argmin2[1]], map_sig_opd[argmin2[2]])
                print('Indexes are:', argmin2)

                log_file = save_path + \
                    '%s_%03d_mapping' % (key, basin_hopping_count) + '.log'
                with open(log_file, 'a') as mlog:
                    mlog.write('******* Mapping %s *******\n' % key)
                    mlog.write(
                        '%s-%02d-%02d %02dh%02d\n' % (datetime.now().year,
                                                      datetime.now().month,
                                                      datetime.now().day,
                                                      datetime.now().hour,
                                                      datetime.now().minute))
                    mlog.write(
                        'Spectral bandwidth (in nm):\n%s,%s\n' % (wl_min,
                                                                  wl_max))
                    mlog.write('Spectral binning %s\n' %
                               activate_spectral_binning)
                    mlog.write('Time binning of photometry %s (%s)\n' % (
                        activate_time_binning_photometry,
                        nb_frames_binning_photometry))
                    mlog.write('Bin boundaries (%s, %s)\n' %
                               (bin_bounds[0], bin_bounds[1]))
                    mlog.write('Number of bins %s\n' % (int(sz**0.5)))
                    mlog.write('Boundaries (na, mu, sig):\n')
                    mlog.write(str(bounds_na)+str(bounds_mu) +
                               str(bounds_sig)+'\n')
                    mlog.write('Number of elements %s\n' % (n_samp_total))
                    mlog.write('Number of loaded points %s (%s, %s)\n' % (
                        data_null.shape[1], *nb_files_data))
                    mlog.write('Number of loaded dark points %s (%s, %s)\n' % (
                        dark['Iminus'].shape[1], *nb_files_dark))
                    mlog.write('Zeta file %s\n' %
                               (os.path.basename(config['zeta_coeff_path'])))
                    mlog.write('Time to load files: %.3f s\n' %
                               (stop_loading - start_loading))
                    mlog.write('Sizes:\n')
                    mlog.write('%s\t%s\t%s\n' %
                               (map_na_sz, map_mu_sz, map_sig_sz))
                    mlog.write('activate phase sorting %s\n' %
                               (activate_phase_sorting))
                    mlog.write('activate_preview_only %s\n' %
                               (activate_preview_only))
                    mlog.write('factor_minus, factor_plus: %s %s\n' % (
                        factor_minus0[idx_null], factor_plus0[idx_null]))
                    mlog.write('nb_frames_sorting_binning %s\n' %
                               (nb_frames_sorting_binning))
                    mlog.write('switch_invert_null %s\n\n' %
                               (switch_invert_null))
                    mlog.write('Results\n')
                    mlog.write(
                        'Min in param space\n%s\t%s\t%s\t%s\n' % (
                            chi2map2.min(), map_na[argmin[0]],
                            map_mu_opd[argmin[1]], map_sig_opd[argmin[2]]))
                    mlog.write('Indexes are\n'+str(argmin)+'\n')
                    mlog.write('2nd min in param space\n%s\t%s\t%s\t%s\n' % (
                        chi2map2[argmin2], map_na[argmin2[0]],
                        map_mu_opd[argmin2[1]], map_sig_opd[argmin2[2]]))
                    mlog.write('Indexes are\n'+str(argmin2)+'\n')
                    mlog.write('Duration: %.3f s\n' % (stop-start))
                    mlog.write('----- End -----\n\n\n')
                    print('Text saved')

                valmin = np.nanmin(chi2map[:, :, :, 0])
                valmax = np.nanmax(chi2map[:, :, :, 0])

                """
                Plot the 3D parameters space

                WARNING: they are in log scale
                """
                if select_optimizer == 0:
                    gff.plot_chi2map([chi2map[i, :, :, 0].T
                                      for i in range(chi2map.shape[0])],
                                     map_mu_opd,
                                     map_sig_opd, map_na, argmin[0], step_mu,
                                     step_sig, 'mu opd', 'sig opd', 'Na', key,
                                     save_path, 'mu', 'sig',
                                     activate_spectral_binning,
                                     basin_hopping_count, wl_min, wl_max, valmin,
                                     valmax)
    
                    gff.plot_chi2map([chi2map[:, :, i, 0]
                                      for i in range(chi2map.shape[2])],
                                     map_mu_opd,
                                     map_na, map_sig_opd, argmin[2], step_mu,
                                     step_na, 'mu opd', 'null depth', 'sig', key,
                                     save_path, 'null', 'mu',
                                     activate_spectral_binning,
                                     basin_hopping_count, wl_min, wl_max, valmin,
                                     valmax)
    
                    gff.plot_chi2map([chi2map[:, i, :, 0]
                                      for i in range(chi2map.shape[1])],
                                     map_sig_opd,
                                     map_na, map_mu_opd, argmin[1], step_sig,
                                     step_na,
                                     'sig opd', 'null depth', 'mu', key, save_path,
                                     'null', 'sig', activate_spectral_binning,
                                     basin_hopping_count, wl_min, wl_max, valmin,
                                     valmax)
                elif select_optimizer == 1 or select_optimizer == 2:
                    gff.plot_lklh_map([chi2map[i, :, :, 0].T
                                      for i in range(chi2map.shape[0])],
                                     map_mu_opd,
                                     map_sig_opd, map_na, argmin[0], step_mu,
                                     step_sig, 'mu opd', 'sig opd', 'Na', key,
                                     save_path, 'mu', 'sig',
                                     activate_spectral_binning,
                                     basin_hopping_count, wl_min, wl_max, valmin,
                                     valmax)
    
                    gff.plot_lklh_map([chi2map[:, :, i, 0]
                                      for i in range(chi2map.shape[2])],
                                     map_mu_opd,
                                     map_na, map_sig_opd, argmin[2], step_mu,
                                     step_na, 'mu opd', 'null depth', 'sig', key,
                                     save_path, 'null', 'mu',
                                     activate_spectral_binning,
                                     basin_hopping_count, wl_min, wl_max, valmin,
                                     valmax)
            else:
                """
                Do MCMC here
                """
                path = save_path
                series = resuts_rsc[0]
                name = resuts_rsc[1] # 1525-1575_AlfBoo
                results_files = [path+key+'_%03d_'%(count)+name+'_pdf.pkl' for count in range(series[0], series[1])]
                
                alfboo_dispersed = gff.Sci(results_files)
                dic_flag = {bl: np.ones(
                    alfboo_dispersed.dic_null[bl].shape[0], dtype=bool) for bl in alfboo_dispersed.dic_null.keys()}
                alfboo_dispersed.flagData(dic_flag)
                alfboo_dispersed.createNullArray(use_chi2=False)                 
                y = alfboo_dispersed.null_measured
                ymu = alfboo_dispersed.mu_measured
                ysig = alfboo_dispersed.sig_measured
                init_pos = np.array([y, ymu, ysig])
                init_pos = np.squeeze(init_pos)
                bounds_fit_reshaped = np.array(bounds_fit).T
                lklh_args = [bounds_fit_reshaped, null_pdf, null_axis, nsc_function]
                if select_optimizer == 1:
                    lklh_func = gff.likelihood
                    lklh_kwargs = None
                elif select_optimizer == 2 or select_optimizer == 0:
                    lklh_func = gff.likelihoodChi2
                    lklh_kwargs = {'data_err': null_pdf_err}
                
                func_kwargs = {'normed': normed, 'verbose': False}
                mcmc_args = resuts_rsc[2]
                print('MCMC: Number of walkers:', mcmc_args[0])
                print('MCMC: Number of steps:', mcmc_args[1])
                
                samples, flat_samples = gff.use_mcmc(lklh_func, lklh_args, init_pos, mcmc_args, lklh_kwargs, func_kwargs)
                ndim = init_pos.size

                np.savez(save_path + key+'_'+str(basin_hopping_count)+'_mcmc_results_'+name, samples=samples, uncertainties=flat_samples.std(0), values=init_pos)
                print('Data savec in: '+save_path+key+'_'+str(basin_hopping_count)+'_mcmc_results_'+name+'.npz')
                print('MCMC uncertainties: ', flat_samples.std(0))
                
                fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
                labels = ["na", "mu", "sig"]
                for i in range(ndim):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                
                axes[-1].set_xlabel("step number")
                fig.tight_layout()
                fig.savefig(save_path +key+'_'+str(basin_hopping_count)+'_mcmc_'+name+'.png', format='png', dpi=150)
                
                fig = corner.corner(
                    flat_samples, labels=['na', 'mu', 'sigma'], truths=init_pos
                )
                fig.savefig(save_path + key+'_'+str(basin_hopping_count)+'_mcmc_corner_plot_'+name+'.png', format='png', dpi=150)
                
            if not chi2_map_switch:
                sys.stdout.close()

            results = {key: [popt_liste, uncertainties_liste, chi2_liste,
                             init_liste, termination_liste, pcov_liste,
                             wl_scale,
                             n_samp_per_loop]}
            wl_scale = wl_scale_saved

            if not skip_fit and not chi2_map_switch and not activate_mcmc:
                '''
                Save the optimal parameters, inputs, fit information of all
                basin hop in one run.
                '''
                pickle_name = key+'_'+'%03d' % (basin_hopping_count)+'_'+str(
                    wl_min)+'-'+str(wl_max)+'_' +\
                    os.path.basename(datafolder[:-1])
                if activate_spectral_binning:
                    pickle_name = pickle_name + '_sb'
                pickle_name = pickle_name+'_pdf'
                pickle_name = pickle_name+'.pkl'

                with open(save_path+pickle_name, 'wb') as f:
                    pickle.dump(results, f)
                    print('--- Pickle saved ---')

    total_time_stop = time()
    print('Total time', total_time_stop-total_time_start)
    plt.ion()
    plt.show()

    print('-- End --')
    
    if activate_preview_only:
        return data, idx_good_frames, i_pm
    elif activate_save_classic_esti:
        return avg, std, wl_scale, data_null
    elif chi2_map_switch:
        return chi2map, map_mu_opd, map_na, map_sig_opd, argmin, data_null,\
            null_axis, null_pdf
    elif activate_mcmc:
        return samples, flat_samples.std(0)
    else:
        return popt[0], uncertainties, chi2, data_null, null_axis, null_pdf,\
            (Iminus, Iplus), data, dark

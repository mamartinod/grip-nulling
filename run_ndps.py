# -*- coding: utf-8 -*-
"""
This script is an example on how to use the library ``nsc.py``
to use the NSC method to get self-calibrated null depth.
"""

from ndps_core import run_ndps
import matplotlib.pyplot as plt
import numpy as np

print('Start')
# =============================================================================
# Settings
# =============================================================================
# Attempt to deconvolve dark noise from photometries by assuming dark
# and photometry to be normally distributed
activate_dark_correction = False
# Activate the frame sorting to remove frames with non_Gaussian phase events
activate_phase_sorting = True
# Use a linear model of the null depth instead of the exact one
activate_linear_model = False
# Oversampling all spectral channels in MC to mimic temporal loss of coherence
activate_oversampling = True
# Display null/antinull outputs vs time for setting the sorting parameters
activate_preview_only = False
# Activate basin-hopping strategy for finding the global minimum in fit
activate_random_init_guesses = True
# Deconvolve photometries after assuming they follow a Normal distribution.
activate_photo_resampling = False
# Calculate the null depth in the old-fashioned way (cf Hanot et al. (2011))
activate_save_classic_esti = False
#
activate_spectral_sorting = False
activate_spectral_binning = False
activate_time_binning_photometry = True
activate_use_antinull = True
activate_use_photometry = False
activate_remove_dark = False
activate_draw_model = True
activate_lbti_mode = False
activate_rvu = False
# Use the measured zeta coeff. If False, value are set to 0.5
activate_zeta = True
# Do not do fit
skip_fit = False
# Explore parameter space instead of fit
chi2_map_switch = False
# Map the parameters space over astronull, DeltaPhi mu and sigma
map_na_sz = 10
map_mu_sz = 80
map_sig_sz = 10
# Binning the frames before any calculation
global_binning = 1
# Total number of elements to generate for the MC
n_samp_total = int(1e+7)
# Number of samples per loop to relieve computation power
n_samp_per_loop = int(1e+7)
# Number of frames to bin before doing the sorting
nb_frames_sorting_binning = 100
# Number of frames to bin to go toward a dark-free histogram of injection
nb_frames_binning_photometry = -1
# Choice of optimizer
select_optimizer = 0 # 0 = Chi2, 1 = lklh
# Series of results
results_series = (1000, 1010)
results_names = '1525-1575_AlfBoo'
# mcmc args
activate_mcmc = True
nwalkers = 6
nstep = 2000
progress_bar = True
mcmc_args = (nwalkers, nstep, progress_bar)
# Which data files to load
# supercount = 1
# z = supercount * 100
# for k in range(z, z+1):
for supercount in range(14, 15):
    plt.close('all')
    z = supercount * 100
    k = z
    nb_files_data = (0, None)
    # Which dark files to load
    nb_files_dark = (0, None)
    # lower and upper bound of the iteration loop for basin hopping method
    basin_hopping_nloop = (10*k, 10*k+10)
    # Baselines to process
    which_nulls = ['null5', 'null6']
    
    # Lower bound of the bandwidth to process
    wl_min = 1525
    # Upper bound of the bandwidth to process
    wl_max = 1575
    
    activates = (activate_dark_correction, activate_phase_sorting, activate_linear_model,
                 activate_oversampling, activate_preview_only,
                 activate_random_init_guesses, activate_photo_resampling,
                 activate_save_classic_esti, activate_spectral_sorting,
                 activate_spectral_binning, activate_time_binning_photometry,
                 activate_use_antinull, activate_use_photometry,
                 activate_zeta, activate_remove_dark, activate_draw_model, activate_lbti_mode,
                 select_optimizer, activate_rvu, activate_mcmc)
    
    maps_sz = (map_na_sz, map_mu_sz, map_sig_sz)
    
    nbs = (global_binning, n_samp_total, n_samp_per_loop,
           nb_frames_sorting_binning,
           nb_frames_binning_photometry, nb_files_data, nb_files_dark,
           basin_hopping_nloop)
    
    wl_minmax = (wl_min, wl_max)
    
    resuts_rsc = (results_series, results_names, mcmc_args)
    
    
    out = run_ndps(activates, skip_fit, chi2_map_switch, maps_sz, nbs, which_nulls,
                  wl_minmax, supercount, resuts_rsc)

# -*- coding: utf-8 -*-
"""
This script is an example on how to use the library ``nsc.py``
to use the NSC method to get self-calibrated null depth.
"""

from ndps_core import run_ndps
import matplotlib.pyplot as plt

# =============================================================================
# Settings
# =============================================================================
# Attempt to deconvolve dark noise from photometries by assuming dark
# and photometry to be normally distributed
activate_dark_correction = False
# Activate the frame sorting to remove frames with non_Gaussian phase events
activate_phase_sorting = False
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
activate_use_antinull = False
activate_use_photometry = False
activate_remove_dark = False
activate_draw_model = True
activate_lbti_mode = True
# Use the measured zeta coeff. If False, value are set to 0.5
activate_zeta = True
# Do not do fit
skip_fit = False
# Explore parameter space instead of fit
chi2_map_switch = False
# Map the parameters space over astronull, DeltaPhi mu and sigma
map_na_sz = 20
map_mu_sz = 100
map_sig_sz = 10
# Binning the frames before any calculation
global_binning = 1
# Total number of elements to generate for the MC
n_samp_total = int(1e+6)
# Number of samples per loop to relieve computation power
n_samp_per_loop = int(1e+6)
# Number of frames to bin before doing the sorting
nb_frames_sorting_binning = 1
# Number of frames to bin to go toward a dark-free histogram of injection
nb_frames_binning_photometry = -1
# Choice of optimizer
select_optimizer = 1 # 0 = Chi2, 1 = lklh

# Which data files to load
# supercount = 1
# z = supercount * 100
# for k in range(z, z+1):
for supercount in range(1, 50):
    plt.close('all')
    z = supercount * 100
    k = z
    nb_files_data = (0, 1000)
    # Which dark files to load
    nb_files_dark = (0, 1000)
    # lower and upper bound of the iteration loop for basin hopping method
    basin_hopping_nloop = (10*k, 10*k+10)
    # Baselines to process
    which_nulls = ['null1']
    
    # Lower bound of the bandwidth to process
    wl_min = 11000
    # Upper bound of the bandwidth to process
    wl_max = 11200
    
    activates = (activate_dark_correction, activate_phase_sorting, activate_linear_model,
                 activate_oversampling, activate_preview_only,
                 activate_random_init_guesses, activate_photo_resampling,
                 activate_save_classic_esti, activate_spectral_sorting,
                 activate_spectral_binning, activate_time_binning_photometry,
                 activate_use_antinull, activate_use_photometry,
                 activate_zeta, activate_remove_dark, activate_draw_model, activate_lbti_mode,
                 select_optimizer)
    
    maps_sz = (map_na_sz, map_mu_sz, map_sig_sz)
    
    nbs = (global_binning, n_samp_total, n_samp_per_loop,
           nb_frames_sorting_binning,
           nb_frames_binning_photometry, nb_files_data, nb_files_dark,
           basin_hopping_nloop)
    
    wl_minmax = (wl_min, wl_max)
    
    
    out = run_ndps(activates, skip_fit, chi2_map_switch, maps_sz, nbs, which_nulls,
                  wl_minmax, supercount)

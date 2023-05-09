#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a configuration file to load different data to fit from GLINT
This file is called by ``nsc.py`` which reads the dictionary
**config**.

See the example in the code to see how to design a configuration.

NOTE: the configuration is encapsulated into a function to make Sphinx happy.
"""
import numpy as np
import os


def prepareConfig():
    # =============================================================================
    #  LBTI data/CAL HD104979
    # =============================================================================
    starname = 'lbti'
    date = '2015-02-08'
    ''' Set the bounds of the parameters to fit '''
    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    bounds_mu0 = [(-2000, 2000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
    bounds_sig0 = [(50, 250), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
    bounds_na0 = [(0.0, 0.018), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    bin_bounds0 = [(-0.05, 0.2), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"

    ''' Set the initial conditions '''
    mu_opd0 = np.array([820., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
    sig_opd0 = np.array([90, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
    na0 = np.array([0.0162, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null

    factor_minus0 = [1., 1, 1, 1., 4.5, 2.5]
    factor_plus0 = [1., 1, 1, 1., 2.5, 2]

    ''' Import real data '''
    datafolder = '2015-02-08_APR/'
    darkfolder = '2015-02-08_APR/'
    root = "/mnt/96980F95980F72D3/lbti/"
    file_path = root+'processed/'+datafolder
    save_path = file_path+'output/'
    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and int(f[15:18]) >= 9 and int(f[15:18]) <= 16]
    dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and int(f[15:18]) >= 9 and int(f[15:18]) <= 16]
    calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    # =============================================================================
    # Set the configuration into a dictionay
    # =============================================================================
    if len(data_list) == 0:
        raise Exception('Data list is empty')
    if len(dark_list) == 0:
        raise Exception('Dark list is empty')

    config = {'nulls_to_invert': nulls_to_invert,
              'nulls_to_invert_model': nulls_to_invert_model,
              'bounds_mu0': bounds_mu0,
              'bounds_sig0': bounds_sig0,
              'bounds_na0': bounds_na0,
              'diffstep': diffstep,
              'xscale': xscale,
              'bin_bounds0': bin_bounds0,
              'mu_opd0': mu_opd0,
              'sig_opd0': sig_opd0,
              'na0': na0,
              'datafolder': datafolder,
              'darkfolder': darkfolder,
              'root': root,
              'file_path': file_path,
              'save_path': save_path,
              'data_list': data_list,
              'dark_list': dark_list,
              'calib_params_path': calib_params_path,
              'zeta_coeff_path': zeta_coeff_path,
              'starname': starname,
              'date': date,
              'factor_minus0': factor_minus0,
              'factor_plus0': factor_plus0}

    return config

if __name__ == '__main__':
    config = prepareConfig()
    import h5py
    a = h5py.File(config['data_list'][0])
    print(np.array(a['Iminus1']).shape)
    print(np.array(a['piston_rms']).shape)    
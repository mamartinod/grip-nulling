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

def prepareConfig(supercount):
    activate_lbti_algo = False
    # datafolder = '2015-02-08_APR/'
    datafolder = '2015-02-08_APR_preprocessed/'
    darkfolder = datafolder
    savefolder = datafolder[:-1]+'_lklh/'
    root = "/mnt/96980F95980F72D3/lbti/"

    #=============================================================================
    #  data202006/AlfBoo/
    #=============================================================================
    starname = 'Alf Boo'
    date = '2020-06-01'
    ''' Set the bounds of the parameters to fit '''
    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 1000), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
    bounds_sig0 = [(100, 300), (200, 300), (200, 300), (10,200), (100, 200), (100, 200)] # bounds for DeltaPhi sig
    bounds_na0 = [(0.0, 0.1), (0., 0.05), (0., 0.01), (0.0, 0.05), (0., 0.05), (0., 0.1)] # bounds for astronull
    diffstep = [0.02, 0.05, 0.05] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    bin_bounds0 = [(-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 1.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"


    ''' Set the initial conditions '''
    mu_opd0 = np.array([510, 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
    sig_opd0 = np.array([8.2, 260, 260, 110, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
    na0 = np.array([0.096, 0.001, 0.001, 0.011, 0.025, 0.08], dtype=np.float64) # initial guess of astro null

    mu_opd0[0] = 3.02383619e+02
    mu_opd0[3] = 4.05482071e+02
    mu_opd0[4] = 9.85782767e+03
    mu_opd0[5] = 1.29190582e+04
    sig_opd0[0] = 1.63288461e+02
    sig_opd0[3] = 1.14174033e+02
    sig_opd0[4] = 1.26271045e+02
    sig_opd0[5] = 1.70226744e+02
    na0[0] = 7.04645603e-02
    na0[3] = 1.10478859e-02
    na0[4] = 2.32493033e-02
    na0[5] = 7.80002694e-02

    factor_minus0 = [1., 1, 1, 1.5, 4.5, 2.5]
    factor_plus0 = [1., 1, 1, 1.5, 2.5, 2]

    ''' Import real data '''
    datafolder = 'data202006/AlfBoo/'
    darkfolder = 'data202006/AlfBoo/'
    # root = "//morgana.physics.usyd.edu.au/Morgana2/snert/"
    root = "C:/Users/marc-antoine/glint/"
    # root = "/mnt/96980F95980F72D3/"
    file_path = root+'GLINTprocessed/'+datafolder
    save_path = file_path+'chi2/'
    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f]
    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark1' in f]
    calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    zeta_coeff_path = calib_params_path + '20200604_zeta_coeff_raw.hdf5'

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
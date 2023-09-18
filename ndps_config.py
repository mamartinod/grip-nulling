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
    savefolder = datafolder[:-1]+'_lklh1/'
    root = "/mnt/96980F95980F72D3/lbti/"

    if supercount == 9:
        # =============================================================================
        #  LBTI data/beta leo  ID009
        # =============================================================================
        ID = 'ID009'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1500), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(150, 350), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([195., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([280, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.007, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 10:
        # =============================================================================
        #  LBTI data/beta leo  ID010
        # =============================================================================
        ID = 'ID010'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 800), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(150, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.25, 0.3, 0.3] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([405., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([225, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.007, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 11:
        # =============================================================================
        #  LBTI data/beta leo ID011
        # =============================================================================
        ID = 'ID011'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-300., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([220, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.007, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 12:
        # =============================================================================
        #  LBTI data/beta leo  ID012
        # =============================================================================
        ID = 'ID012'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([300., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([340, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.005, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 13:
        # =============================================================================
        #  LBTI data/beta leo  ID013
        # =============================================================================
        ID = 'ID013'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([300., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([250, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.005, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'
    
    if supercount == 14:
        # =============================================================================
        #  LBTI data/beta leo  ID014
        # =============================================================================
        ID = 'ID014'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0., 0.12), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([480., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([260, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.006, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 15:
        # =============================================================================
        #  LBTI data/beta leo  ID015
        # =============================================================================
        ID = 'ID015'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0., 0.12), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([290., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([280, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.007, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 16:
        # =============================================================================
        #  LBTI data/beta leo  ID016
        # =============================================================================
        ID = 'ID016'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0., 0.12), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([370., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([320, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.009, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 26:
        # =============================================================================
        #  LBTI data/beta leo  ID026
        # =============================================================================
        ID = 'ID026'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(300, 500), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.2), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([390., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([420, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.011, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 27:
        # =============================================================================
        #  LBTI data/beta leo  ID027
        # =============================================================================
        ID = 'ID027'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(350, 550), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0., 0.12), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([90., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([420, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.011, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'
 
    if supercount == 28:
        # =============================================================================
        #  LBTI data/beta leo  ID028 - bad fit (no longer)
        # =============================================================================
        ID = 'ID028'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-100, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.025), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0.0, 0.08), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-89., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([300, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.01, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 29:
        # =============================================================================
        #  LBTI data/beta leo  ID029
        # =============================================================================
        ID = 'ID029'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([310., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([360, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.006, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 30:
        # =============================================================================
        #  LBTI data/beta leo  ID030
        # =============================================================================
        ID = 'ID030'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(500, 700), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0., 0.18), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([240., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([580, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.015, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 31:
        # =============================================================================
        #  LBTI data/beta leo  ID031
        # =============================================================================
        ID = 'ID031'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(300, 500), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.2), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([660., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([420, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.006, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 32:
        # =============================================================================
        #  LBTI data/beta leo  ID032
        # =============================================================================
        ID = 'ID032'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0., 0.15), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([410., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([280, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.007, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 33:
        # =============================================================================
        #  LBTI data/beta leo  ID033
        # =============================================================================
        ID = 'ID033'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(400, 600), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(0.0, 0.02), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0., 0.15), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([110., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([500, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.007, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'SCI' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'SCI' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 1:
        # =============================================================================
        #  LBTI data/HD104979 ID001
        # =============================================================================
        ID = 'ID001'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.005, 0.057), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-220., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([160, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.00, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 2:
        # =============================================================================
        #  LBTI data/HD104979 ID002
        # =============================================================================
        ID = 'ID002'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.005, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([420., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([220, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 3:
        # =============================================================================
        #  LBTI data/HD104979 ID003
        # =============================================================================
        ID = 'ID003'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(300, 500), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.015, 0.005), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.005, 0.08), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([80., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([360, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([-0.001, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 4:
        # =============================================================================
        #  LBTI data/HD104979 ID004
        # =============================================================================
        ID = 'ID004'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(300, 500), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.005, 0.15), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([420., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([380, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 5:
        # =============================================================================
        #  LBTI data/HD104979 ID005
        # =============================================================================
        ID = 'ID005'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(300, 500), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.005, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([120., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([360, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.001, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'
    
    if supercount == 6:
        # =============================================================================
        #  LBTI data/HD104979 ID006
        # =============================================================================
        ID = 'ID006'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.005, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([420., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([220, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 7:
        # =============================================================================
        #  LBTI data/HD104979 ID007
        # =============================================================================
        ID = 'ID007'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.005, 0.08), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-180., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([280, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.001, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 8:
        # =============================================================================
        #  LBTI data/HD104979 ID008
        # =============================================================================
        ID = 'ID008'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.005, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([360., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([200, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.004, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 17:
        # =============================================================================
        #  LBTI data/HD104979 ID017
        # =============================================================================
        ID = 'ID017'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(10, 210), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.02, 0.06), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([140., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([70, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 18:
        # =============================================================================
        #  LBTI data/HD104979 ID018 - bad fit
        # =============================================================================
        ID = 'ID018'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.02, 0.08), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-80., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([240, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.004, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 19:
        # =============================================================================
        #  LBTI data/HD104979 ID019
        # =============================================================================
        ID = 'ID019'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(1, 201), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.06), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-260., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([81, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.001, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 20:
        # =============================================================================
        #  LBTI data/HD104979 ID020
        # =============================================================================
        ID = 'ID020'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.08), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-180., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([240, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.005, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        if activate_lbti_algo:
            save_path = root+'processed/IDL/'+datafolder+ID+'/'
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 21:
        # =============================================================================
        #  LBTI data/HD104979 ID021
        # =============================================================================
        ID = 'ID021'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.08), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-160., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([160, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.009, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 22:
        # =============================================================================
        #  LBTI data/HD104979 ID022
        # =============================================================================
        ID = 'ID022'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.06), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([0., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([180, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 23:
        # =============================================================================
        #  LBTI data/HD104979 ID023
        # =============================================================================
        ID = 'ID023'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(400, 600), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.12), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-40., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([480, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 24:
        # =============================================================================
        #  LBTI data/HD104979 ID024
        # =============================================================================
        ID = 'ID024'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(50, 200), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.06), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-200., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([110, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.004, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 25:
        # =============================================================================
        #  LBTI data/HD104979 ID025
        # =============================================================================
        ID = 'ID025'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(300, 500), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.11), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([40., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([480, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.004, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 34:
        # =============================================================================
        #  LBTI data/HD104979 ID034
        # =============================================================================
        ID = 'ID034'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.11), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-300., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([300, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 35:
        # =============================================================================
        #  LBTI data/HD104979 ID035
        # =============================================================================
        ID = 'ID035'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.15), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([500., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([320, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 36:
        # =============================================================================
        #  LBTI data/HD104979 ID036
        # =============================================================================
        ID = 'ID036'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0., 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-240., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([360, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.006, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 37:
        # =============================================================================
        #  LBTI data/HD104979 ID037
        # =============================================================================
        ID = 'ID037'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0., 0.2), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-660., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([340, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.001, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 38:
        # =============================================================================
        #  LBTI data/HD104979 ID038 - bad fit (not anymore)
        # =============================================================================
        ID = 'ID038'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.15), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([420., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([360, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 39:
        # =============================================================================
        #  LBTI data/HD104979 ID039 - bad fit (no longer)
        # =============================================================================
        ID = 'ID039'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-360., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([240, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.001, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 40:
        # =============================================================================
        #  LBTI data/HD104979 ID040
        # =============================================================================
        ID = 'ID040'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0.0, 0.12), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([60., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([340, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.005, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        # save_path = file_path+ID+'/'
        save_path = root+'processed/IDL/'+datafolder+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 41:
        # =============================================================================
        #  LBTI data/HD104979 ID041
        # =============================================================================
        ID = 'ID041'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0.0, 0.12), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-420., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([260, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 42:
        # =============================================================================
        #  LBTI data/HD104979 ID042
        # =============================================================================
        ID = 'ID042'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0.0, 0.13), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-420., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([260, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 43:
        # =============================================================================
        #  LBTI data/HD104979 ID043
        # =============================================================================
        ID = 'ID043'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0.0, 0.13), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-420., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([320, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.004, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 44:
        # =============================================================================
        #  LBTI data/HD104979 ID044
        # =============================================================================
        ID = 'ID044'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-1000, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(0.0, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([360., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([180, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.001, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 45:
        # =============================================================================
        #  LBTI data/HD104979 ID045
        # =============================================================================
        ID = 'ID045'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(100, 300), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([120., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([240, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 46:
        # =============================================================================
        #  LBTI data/HD104979 ID046
        # =============================================================================
        ID = 'ID046'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-500, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([55., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([320, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.005, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 47:
        # =============================================================================
        #  LBTI data/HD104979 ID047
        # =============================================================================
        ID = 'ID047'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-500, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(200, 400), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-215., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([260, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.003, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 48:
        # =============================================================================
        #  LBTI data/HD104979 ID048
        # =============================================================================
        ID = 'ID048'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-500, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(10, 210), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-125., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([150, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([0.008, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'

    if supercount == 49:
        # =============================================================================
        #  LBTI data/HD104979 ID049
        # =============================================================================
        ID = 'ID049'
        starname = 'lbti'
        date = '2015-02-08'
        ''' Set the bounds of the parameters to fit '''
        nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
        nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
        bounds_mu0 = [(-500, 1000), (2200, 2500), (2200, 2500), (0, 800), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
        bounds_sig0 = [(10, 210), (200, 300), (200, 300), (70,170), (100, 200), (100, 200)] # bounds for DeltaPhi sig
        bounds_na0 = [(-0.01, 0.01), (0., 0.05), (0., 0.01), (0.0, 0.005), (0., 0.05), (0., 0.1)] # bounds for astronull
        diffstep = [0.005, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
        xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
        bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 2.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
        ''' Set the initial conditions '''
        mu_opd0 = np.array([-380., 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
        sig_opd0 = np.array([190, 260, 260, 150, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
        na0 = np.array([-0.001, 0.001, 0.001, 0.002, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    
        factor_minus0 = [2., 1, 1, 1., 4.5, 2.5]
        factor_plus0 = [1., 1, 1, 1., 2.5, 2]
    
        ''' Import real data '''
        # datafolder = '2015-02-08_APR/'
        # darkfolder = '2015-02-08_APR/'
        root = "/mnt/96980F95980F72D3/lbti/"
        file_path = root+'processed/'+datafolder
        save_path = file_path+ID+'/'
        data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'NULL' in f and 'CAL' in f and ID in f]
        dark_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'BCKG' in f and 'CAL' in f and ID in f]
        calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
        zeta_coeff_path = calib_params_path + 'lbti_zeta_coeff_raw.hdf5'


    # =============================================================================
    # Set the configuration into a dictionay
    # =============================================================================
    save_path = root+'processed/'+savefolder+ID+'/'
    if activate_lbti_algo:
        save_path = root+'processed/IDL/'+savefolder+ID+'/'

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
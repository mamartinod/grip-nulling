#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module gathers functions to read files.
"""

import numpy as np
import h5py
from astropy.io import fits

def load_data(data_batch, kw_to_extract, wl_edges, file_type, *args):
    """
    Load the data from HDF5 or FITS file, select the desired keywords and store the selected data\
        in a dictionary.

    Parameters
    ----------
    data_batch : list
        Sequence of hdf5 files.
    kw_to_extract : list
        Sequence of keywords to extract from the data files.
    wl_edges : 2-tuple
        Minimum and maximum values of the wavelength to keep.
    file_type : string
        Type of files to load. Accept only 'hdf5' for HDF5 files \
            and 'fits' for FITS files.
    args : int
        If a FITS file is load, it indicates which HDU to load. 
        If HDF5, it does not have to be defined.

    Returns
    -------
    data_dic : dict
        Dictionary containing the loaded data.

    """

    wl_scale = []
    data_dic = {}
    
    try:
        kw_to_extract.remove('wl_scale')
    except ValueError:
        pass

    if file_type.lower() == 'hdf5':
        data_dic, wl_scale = _load_hdf5(data_batch, kw_to_extract, wl_scale, data_dic)
    elif file_type.lower() == 'fits':
        if len(args) != 1:
            raise TypeError('Loading FITS file requires *args of length 1 and having the index of the HDU where to pick data.')
        data_dic, wl_scale = _load_fits(data_batch, kw_to_extract, wl_scale, data_dic, args[0])
    else:
        raise TypeError('file_type must be equal to "hdf5" or "fits".')
        

    # All the wl scale are supposed to be the same, just pick up the first of the list
    wl_scale = wl_scale[0]
    wl_sz = wl_scale.size
    mask = np.arange(wl_sz)

    wl_min, wl_max = wl_edges
    mask = mask[(wl_scale >= wl_min) & (wl_scale <= wl_max)]

    # Merge data along frame axis and trim the wavelengths
    for key in data_dic.keys():
        temp = data_dic[key]
        temp = [selt for elt in temp for selt in elt]
        temp = np.array(temp)
        if temp.ndim > 1:
            temp = temp[:, mask]
        temp = temp.T # Put the wavelength axis first
        data_dic[key] = temp
    
    wl_scale = wl_scale[mask]
    data_dic['wl_scale'] = wl_scale

    return data_dic

def _load_hdf5(data_batch, kw_to_extract, wl_scale, data_dic):
    """
    Load HDF5 files and extract some data then put them in a dictionary.

    Parameters
    ----------
    data_batch : list
        Sequence of hdf5 files.
    kw_to_extract : list
        Sequence of keywords to extract from the data files.
    wl_scale : list
        List to store the wavelengths scales.
    data_dic : dict
        Dictionary storing the data.

    Returns
    -------
    data_dic : dict
        Dictionary storing the data, which was taken as input.
    wl_scale : TYPE
        List to store the wavelengths scales, which was taken as input.

    """
    for d in data_batch:
        print(d)
        with h5py.File(d, 'r') as data_file:
            try:
                wl_scale.append(np.array(data_file['wl_scale']))
            except KeyError:
                print('Wl scale not found, check the keyword "wl_scale" exists in the file')
                break

            for key in kw_to_extract:
                if key in data_dic:
                    data_dic[key].append(np.array(data_file[key]))
                else:
                    data_dic[key] = [np.array(data_file[key])]
                    
    return data_dic, wl_scale

def _load_fits(data_batch, kw_to_extract, wl_scale, data_dic, idx_hdu):
    """
    Load FITS files and extract some data then put them in a dictionary.

    Parameters
    ----------
    data_batch : list
        Sequence of hdf5 files.
    kw_to_extract : list
        Sequence of keywords to extract from the data files.
    wl_scale : list
        List to store the wavelengths scales.
    data_dic : dict
        Dictionary storing the data.
    idx_hdu : int
        Index of the HDU to load.

    Returns
    -------
    data_dic : dict
        Dictionary storing the data, which was taken as input.
    wl_scale : TYPE
        List to store the wavelengths scales, which was taken as input.

    """
    for d in data_batch:
        print(d)
        hdu = fits.open(d)
        hdu_data = hdu[idx_hdu].data
        try:
            wl_scale.append(hdu_data['wl_scale'])
        except KeyError:
            print('Wl scale not found, check the keyword "wl_scale" exists in the file')
            break

        # To find the spectral axis
        wl_sz = hdu_data['wl_scale'].size
        for key in kw_to_extract:
            hdu_data = hdu_data[key]
            
            # Find the spectral axis
            wl_ax = np.where(np.array(hdu_data.shape) == wl_sz)[0][0]
            
            # We must put it on the last axis
            if wl_ax == 0:
                hdu_data = hdu_data.T # Assume hdu_data.ndim = 2
                
            if key in data_dic:
                data_dic[key].append(hdu_data)
            else:
                data_dic[key] = [hdu_data]
                    
    return data_dic, wl_scale

    
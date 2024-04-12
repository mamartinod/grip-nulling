#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module gathers different and unrelated functions.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def get_zeta_coeff(path, wl_scale, plot=False, **kwargs):
    """
    Interpolate the zeta coefficients for the requested wavelengths.

    Parameters
    ----------
    path : string
        Path to the zeta coefficients' file.
    wl_scale : array
        List of wavelength for which we want the zeta coefficients.
    plot : bool, optional
        If `True`, the plot of the interpolated zeta coefficients\
            curve is displayed. The default is False.
    **kwargs : extra keyword arguments
        `wl_bounds` prunes the zeta coeff arrays for them to all have\
        the same wavelength scale.

    Returns
    -------
    coeff_new : dict
        Dictionary of the interpolated zeta coefficients.

    """
    coeff_new = {}
    with h5py.File(path, 'r') as coeff:
        wl = np.array(coeff['wl_scale'])[::-1]
        if 'wl_bounds' in kwargs:  # Average zeta coeff in the bandwidth
            wl_bounds = kwargs['wl_bounds']
            wl_scale = wl[(wl >= wl_bounds[0]) & (wl <= wl_bounds[1])]
        else:
            pass

        for key in coeff.keys():
            if 'wl_bounds' in kwargs:  # Average zeta coeff in the bandwidth
                if key != 'wl_scale':
                    interp_zeta = np.interp(
                        wl_scale[::-1], wl, np.array(coeff[key])[::-1])
                    coeff_new[key] = np.array([np.mean(interp_zeta[::-1])])
                else:
                    coeff_new[key] = np.array([wl_scale.mean()])
            else:
                if key != 'wl_scale':
                    interp_zeta = np.interp(
                        wl_scale[::-1], wl, np.array(coeff[key])[::-1])
                    coeff_new[key] = interp_zeta[::-1]
                else:
                    coeff_new[key] = wl_scale
        if plot:
            plt.figure()
            plt.plot(np.array(coeff['wl_scale']),
                     np.array(coeff['b1null1']), 'o-')
            plt.plot(coeff_new['wl_scale'], coeff_new['b1null1'], '+-')
            plt.grid()
            plt.ylim(-1, 10)

    return coeff_new


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

    
class Logger(object):
    """Save the content of the console inside a txt file.

    Class allowing to save the content of the console inside a txt file.
    Found on the internet, source lost.
    
    To stop the log in the file, use the command `sys.stdout.close()`.
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
        
def get_injection_and_spectrum(photoA, photoB, wl_scale,
                               wl_bounds):
    """
    Get the distributions of the broadband injections and the spectra of\
        beams A and B.

    Parameters
    ----------
    photoA : array-like
        Values of the photometric output of beam A.
    photoB : array-like
        Values of the photometric output of beam B.
    wl_scale : array-like
        Wavelength of the spectra in nm.
    wl_bounds : 2-tuple, optional
        Boundaries between which the spectra are extracted.\
            The wavelengths are expressed in nm.

    Returns
    -------
    2-tuple of 2d-array
        The first element contains the broadband\
                injection of beams A and B, respectively. The second element\
                contains the spectra of beams A and B, respectively.

    """
    # Select the large bandwidth on which we measure the injection
    idx_wl = np.arange(wl_scale.size)
    idx_wl = idx_wl[(wl_scale >= wl_bounds[0]) & (wl_scale <= wl_bounds[1])]
    photoA = photoA[idx_wl]
    photoB = photoB[idx_wl]

    # Extract the spectrum
    spectrumA = photoA.mean(axis=1)
    spectrumB = photoB.mean(axis=1)
    spectrumA = spectrumA / spectrumA.sum()
    spectrumB = spectrumB / spectrumB.sum()

    # Extract the injection for generating random values
    fluxA = photoA.sum(axis=0)
    fluxB = photoB.sum(axis=0)
    
    fluxes = np.array([fluxA, fluxB])
    spectra = np.array([spectrumA, spectrumB])

    return (fluxes, spectra)


def check_init_guess(guess, l_bound, u_bound):
    """
    Check the initial guess in config file are between the bounds for a\
        parameter to fit.

    Parameters
    ----------
    guess : float
        value of the initial guess in config file.
    l_bound : float
        value of the lower bound in config file.
    u_bound : float
        value of the upper bound in config file.

    Returns
    -------
    check : bool
        ``True`` if the initial guess is not between the bounds.

    """
    check = np.any(guess <= np.array(u_bound)[:, 0]) or np.any(
        guess >= np.array(l_bound)[:, 1])
    return check

def return_neg_func(func):
    """
    Return a callable which is the negative of a function: `f(x) -> -f(x)`.
    
    It can be used to create a callable cost function one wants to minimize (e.g. :math:`\chi^2` estimator).

    Parameters
    ----------
    func : callable
        function to return the negative version.

    Returns
    -------
    callable
        negative version of the function.

    """
    def wrapper(*args, **kwargs):
        return -func(*args, **kwargs)
    return wrapper

def tempering(func, tempering_factor):
    """
    Scale a function by a constant.
    It performs *tempering* in MCMC, i.e. to smoothen/sharpen the log-likelihood function.
    Indeed, if the log-likelihood decrease by 1 unit, it means the event is 2.7x less likely to happen.
    Some log-likelihood functions needs to be tempered before being explored by MCMC algorithm.
    
    Note: the shape of the posterior is scaled by the square root of the tempering factor :math:`1 / \sqrt{tempering~factor}`.
        
    Parameters
    ----------
    func : callable
        Function to rescale.
    tempering_factor : float
        Scale factor.

    Returns
    -------
    callable
        Rescaled function.

    >>> tempering(log_chi2, -2 / ddof) # Returns a reduced chi2 cost function

    """
    def tempered_func(*args, **kwargs):
        return func(*args, **kwargs) * tempering_factor
    return tempered_func

# def load_data(data_batch, kw_to_extract, wl_edges):
#     """
#     DEPRECATED
    
#     Load the data from HDF5 file format, select the desired keywords and store the selected data\
#         in a dictionary.

#     Parameters
#     ----------
#     data_batch : list
#         Sequence of hdf5 files.
#     wl_edges : 2-tuple
#         Minimum and maximum values of the wavelength to keep.
#     kw_to_extract : list
#         Sequence of keywords to extract from the HDF5 files.

#     Returns
#     -------
#     data_dic : dict
#         Dictionary containing the loaded data.

#     """

#     wl_scale = []
#     data_dic = {}
    
#     try:
#         kw_to_extract.remove('wl_scale')
#     except ValueError:
#         pass

#     for d in data_batch:
#         print(d)
#         with h5py.File(d, 'r') as data_file:
#             try:
#                 wl_scale.append(np.array(data_file['wl_scale']))
#             except KeyError:
#                 print('Wl scale not found, check the keyword "wl_scale" exists in the file')
#                 break

#             for key in kw_to_extract:
#                 if key in data_dic:
#                     data_dic[key].append(np.array(data_file[key]))
#                 else:
#                     data_dic[key] = [np.array(data_file[key])]


#     # All the wl scale are supposed to be the same, just pick up the first of the list
#     wl_scale = wl_scale[0]
#     mask = np.arange(wl_scale.size)

#     wl_min, wl_max = wl_edges
#     mask = mask[(wl_scale >= wl_min) & (wl_scale <= wl_max)]

#     # Merge data along frame axis and trim the wavelengths
#     for key in data_dic.keys():
#         temp = data_dic[key]
#         temp = [selt for elt in temp for selt in elt]
#         temp = np.array(temp)
#         if temp.ndim > 1:
#             temp = temp[:, mask]
#         temp = temp.T # Put the wavelength axis first
#         data_dic[key] = temp
    
#     wl_scale = wl_scale[mask]
#     data_dic['wl_scale'] = wl_scale

#     return data_dic
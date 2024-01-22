#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module gathers different and unrelated functions.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys


def get_zeta_coeff(path, wl_scale, plot=False, **kwargs):
    """Interpolate the zeta coefficients for the requested wavelengths.

    :param path: Path to the zeta coefficients' file.
    :type path: string
    :param wl_scale: List of wavelength for which we want the zeta\
        coefficients.
    :type wl_scale: array
    :param plot: If ``True``, the plot of the interpolated zeta coefficients\
        curve is displayed, defaults to False
    :type plot: bool, optional
    :param **kwargs: Bins the zeta coefficient between the specified\
        wavelength in this keyword.
    :type **kwargs: extra keyword arguments
    :return: Dictionary of the interpolated zeta coefficients.
    :rtype: dict

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

def load_data(data, wl_edges, kw_to_extract):

    wl_scale = []
    data_dic = {}
    
    try:
        kw_to_extract.remove('wl_scale')
    except ValueError:
        pass

    for d in data:
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


    # All the wl scale are supposed to be the same, just pick up the first of the list
    wl_scale = wl_scale[0]
    mask = np.arange(wl_scale.size)

    wl_min, wl_max = wl_edges
    mask = mask[(wl_scale >= wl_min) & (wl_scale <= wl_max)]

    # Merge data along frame axis and trim the wavelengths
    for key in data_dic.keys():
        temp = data_dic[key]
        temp = [selt for elt in temp for selt in elt]
        temp = np.array(temp)
        print(key, temp.shape)
        if temp.ndim > 1:
            temp = temp[:, mask]
        temp = temp.T # Put the wavelength axis first
        data_dic[key] = temp
    
    wl_scale = wl_scale[mask]
    data_dic['wl_scale'] = wl_scale

    return data_dic

class Logger(object):
    """Save the content of the console inside a txt file.

    Class allowing to save the content of the console inside a txt file.
    Found on the internet, source lost.
    
    To stop the log in the file, use the command ``sys.stdout.close()``.
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
                               wl_bounds=(1400, 1700)):
    """Get the distributions of the broadband injections and the spectra of\
        beams A and B.

    :param photoA: Values of the photometric output of beam A.
    :type photoA: array-like
    :param photoB: Values of the photometric output of beam B.
    :type photoB: array-like
    :param wl_scale: Wavelength of the spectra in nm.
    :type wl_scale: array-like
    :param wl_bounds: Boundaries between which the spectra are extracted.
        The wavelengths are expressed in nm, defaults to (1400, 1700)
    :type wl_bounds: 2-tuple, optional
    :return: The first tuple contains the histograms of the broadband
            injection of beams A and B, respectively. The second tuple
            contains the spectra of beams A and B, respectively.
    :rtype: 2-tuple of 2-tuple

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

    return (fluxes, (spectrumA, spectrumB))





def check_init_guess(guess, l_bound, u_bound):
    """Check the initial guess in config file are between the bounds for a\
        parameter to fit.

    :param guess: value of the initial guess in config file
    :type guess: float
    :param l_bound: value of the lower bound in config file
    :type l_bound: float
    :param u_bound: value of the upper bound in config file
    :type u_bound: float
    :return: ``True`` if the initial guess is not between the bounds.
    :rtype: bool

    """
    check = np.any(guess <= np.array(u_bound)[:, 0]) or np.any(
        guess >= np.array(l_bound)[:, 1])
    return check

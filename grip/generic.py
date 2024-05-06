#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module gathers different and unrelated functions.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
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




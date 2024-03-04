#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains models to perform NSC on GLINT Mark II (2019-2021) and LBTI nuller.

How to build a model: 
    - 1st argument = null depth, 
    - 2nd = wavelength scale
    - 3rd = int representing the iteration over wavelength scale
    - next: "constants" values (not random values for MC)
    - end: quantities generated by MC

"""
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp

def glint_model(na, wavelength, wl_idx, spec_chan_width, spectra, zeta_minus_A, zeta_minus_B,\
                zeta_plus_A, zeta_plus_B, opd, injA, injB,\
                dark_null, dark_antinull):
    """
    Model of the GLINT instrument (version of the instrument 2019 - 2021).

    Parameters
    ----------
    na : float
        Self-calibrated null depth.
    wavelength : 1d-array
        Wavelength in the same unit as the OPD.
    wl_idx : int
        wavelength cursor.
    spec_chan_width : float
        Width of a spectral channel, in the same unit as ``wavelength``.
    spectra : 2d-array
        Spectrum of the source seen by both aperture.
    zeta_minus_A : 1d-array
        Splitting ratio between the null output and the photometric output of beam A.
    zeta_minus_B : 1d-array
        Splitting ratio between the null output and the photometric output of beam B.
    zeta_plus_A : 1d-array
        Splitting ratio between the anti-null output and the photometric output of beam A.
    zeta_plus_B : 1d-array
        Splitting ratio between the anti-null output and the photometric output of beam B.
    opd : 1d-array
        Sequence of OPD.
    injA : 1d-array
        Sequence of injection of beam A.
    injB : 1d-array
        Sequence of injection of beam B.
    dark_null : 1d-array
        Detector noise in the null output.
    dark_antinull : 1d-array
        Detector noise in the anti-null output.

    Returns
    -------
    null : 1d-array
        Sequence of simulated null depth.
    Iminus : 1d-array
        Sequence of simulated null output.
    Iplus : 1d-array
        Sequence of simulated anti-null output.

    """
    visibility = (1 - na) / (1 + na)
    wave_number = 1./wavelength
    sine = cp.sin(2*np.pi*wave_number*(opd))

    delta_wave_number = abs(
        1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
    arg = np.pi*delta_wave_number * (opd)
    sinc = cp.sin(arg) / arg
    sine = sine * sinc

    IA = injA * spectra[0][wl_idx]
    IB = injB * spectra[1][wl_idx]

    Iminus = IA*zeta_minus_A[wl_idx] + IB*zeta_minus_B[wl_idx] - \
    2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A[wl_idx] *
                                    zeta_minus_B[wl_idx]) * visibility * sine

    Iplus = IA*zeta_plus_A[wl_idx] + IB*zeta_plus_B[wl_idx] + \
        2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A[wl_idx]*zeta_plus_B[wl_idx]) * \
        visibility * sine 

    Iminus = Iminus + dark_null
    Iplus = Iplus + dark_antinull
    null = Iminus / Iplus
    return null, Iminus, Iplus
        
def lbti_model(na, wavelength, wl_idx, spec_chan_width, phase_bias, 
               opd, IA, IB, thermal_bckg, sigma_eps):
    """Compute the null depth.

    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the ratio of the null over the antinull fluxes.



    """
    visibility = (1 - na) / (1 + na)
    wave_number = 1./wavelength
    cosine = cp.cos(2 * np.pi * wave_number * opd + phase_bias)
    # delta_wave_number = abs(
    #     1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
    delta_wave_number = spec_chan_width / wavelength**2
    arg = np.pi*delta_wave_number * opd
    sinc = cp.sin(arg) / arg

    cosine = cosine * sinc

    blurring = (1 - 0.5*sigma_eps**2 + 0.125 * sigma_eps**4)
    Iminus = IA + IB + 2 * np.sqrt(IA * IB) * visibility * blurring * cosine
    Iplus = IA + IB + 2 * np.sqrt(IA * IB)
    Iminus = Iminus + thermal_bckg
    null = Iminus / Iplus

    return null, Iminus, Iplus
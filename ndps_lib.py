#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library of the ``nsc.py``.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import h5py
import os
from cupyx.scipy.special import ndtr
from scipy.stats import norm
from scipy.linalg import svd
import warnings
from scipy.optimize import OptimizeWarning, minimize, least_squares

interpolate_kernel = cp.ElementwiseKernel(
    'float32 x_new, raw float32 xp, int32 xp_size, raw float32 yp',
    'raw float32 y_new',

    '''
    int high = xp_size - 1;
    int low = 0;
    int mid = 0;

    while(high - low > 1)
    {
        mid = (high + low)/2;

        if (xp[mid] <= x_new)
        {
            low = mid;
        }
        else
        {
            high = mid;
        }
    }
    y_new[i] = yp[low] + (x_new - xp[low])  * (yp[low+1] - yp[low]) / (xp[low+1] - xp[low]);

    if (x_new < xp[0])
    {
         y_new[i] = yp[0];
    }
    else if (x_new > xp[xp_size-1])
    {
         y_new[i] = yp[xp_size-1];
    }

    '''
)

computeCdfCuda = cp.ElementwiseKernel(
    'float32 x_axis, raw float32 rv, float32 rv_sz',
    'raw float32 cdf',
    '''
    int low = 0;
    int high = rv_sz;
    int mid = 0;

    while(low < high){
        mid = (low + high) / 2;
        if(rv[mid] <= x_axis){
            low = mid + 1;
        }
        else{
            high = mid;
        }
    }
    cdf[i] = high;
    '''
)


def computeCdf(absc, data, mode, normed):
    """Compute the empirical cumulative density function (CDF) on GPU\
        with CUDA.

    :param absc: Abscissa of the CDF.
    :type absc: cupy array
    :param data: Data used to create the CDF.
    :type data: cupy array
    :param mode: If ``ccdf``, the survival function (complementary of the CDF)\
        is calculated instead.
    :type mode: string
    :param normed: If ``True``, the CDF is normed so that the maximum is\
        equal to 1.
    :type normed: bool
    :return: CDF of **data**.
    :rtype: cupy array

    """
    cdf = cp.zeros(absc.shape, dtype=cp.float32)
    data = cp.asarray(data, dtype=cp.float32)
    absc = cp.asarray(absc, dtype=cp.float32)

    data = cp.sort(data)

    computeCdfCuda(absc, data, data.size, cdf)

    if mode == 'ccdf':
        cdf = data.size - cdf

    if normed:
        cdf = cdf/data.size

    return cdf


def rv_generator_wPDF(bins_cent, pdf, nsamp):
    """Random values generator based on the PDF.

    :param bins_cent: Centered bins of the PDF.
    :type bins_cent: array
    :param pdf: Normalized arbitrary PDF to use to generate rv.
    :type pdf: array
    :param nsamp: Number of values to generate.
    :type nsamp: int
    :return: Array of random values generated from the PDF.
    :rtype: array

    """
    bin_width = bins_cent[1] - bins_cent[0]
    cdf = cp.cumsum(pdf, dtype=cp.float32) * bin_width
    cdf, mask = cp.unique(cdf, True)

    cdf_bins_cent = bins_cent[mask]
    cdf_bins_cent = cdf_bins_cent + bin_width/2.

    rv_uniform = cp.random.rand(nsamp, dtype=cp.float32)
    output_samples = cp.zeros(rv_uniform.shape, dtype=cp.float32)
    interpolate_kernel(rv_uniform, cdf, cdf.size,
                       cdf_bins_cent, output_samples)

    return output_samples


def rv_generator(absc, cdf, nsamp, rvu=None):
    """Random values generator based on the CDF.

    :param absc: Abscissa of the CDF;
    :type absc: cupy array
    :param cdf: Normalized arbitrary CDF to use to generate rv.
    :type cdf: cupy array
    :param nsamp: Number of values to generate.
    :type nsamp: int
    :return: Array of random values generated from the CDF.
    :rtype: array

    """
    cdf, mask = cp.unique(cdf, True)
    cdf_absc = absc[mask]

    if rvu is None:
        rv_uniform = cp.random.rand(nsamp, dtype=cp.float32)
    else:
        rv_uniform = cp.array(rvu, dtype=cp.float32)
    output_samples = cp.zeros(rv_uniform.shape, dtype=cp.float32)
    interpolate_kernel(rv_uniform, cdf, cdf.size, cdf_absc, output_samples)

    return output_samples


def computeCdfCpu(rv, x_axis, normed=True):
    """Compute the empirical cumulative density function (CDF) on CPU.

    :param rv: data used to compute the CDF.
    :type rv: array
    :param x_axis: Abscissa of the CDF.
    :type x_axis: array
    :param normed:  If ``True``, the CDF is normed so that the maximum is\
        equal to 1., defaults to True
    :type normed: bool, optional
    :return: CDF of the **data**,  Indexes of cumulated values.
    :rtype: tuple

    """
    cdf = np.ones(x_axis.size)*rv.size
    temp = np.sort(rv)
    idx = 0
    for i in range(x_axis.size):
        #        idx = idx + len(np.where(temp[idx:] <= x_axis[i])[0])
        mask = np.where(temp <= x_axis[i])[0]
        idx = len(mask)

        if len(temp[idx:]) != 0:
            cdf[i] = idx
        else:
            print('pb', i, idx)
            break

    if normed:
        cdf /= float(rv.size)
        return cdf
    else:
        return cdf, mask


def computeCdfCupy(rv, x_axis):
    """Compute the empirical cumulative density function (CDF) on GPU\
        with cupy.

    :param rv: Data used to compute the CDF.
    :type rv: array
    :param x_axis: Abscissa of the CDF.
    :type x_axis: array
    :return: CDF of **data**.
    :rtype: array

    """
    cdf = cp.ones(x_axis.size, dtype=cp.float32)*rv.size
    temp = cp.asarray(rv, dtype=cp.float32)
    temp = cp.sort(rv)
    idx = 0
    for i in range(x_axis.size):
        idx = idx + len(cp.where(temp[idx:] <= x_axis[i])[0])

        if len(temp[idx:]) != 0:
            cdf[i] = idx
        else:
            break

    cdf = cdf / rv.size

    return cdf


def load_data(data, wl_edges, null_key, nulls_to_invert, *args, **kwargs):
    """Load data from data file to create the histograms of the null depths\
        and do Monte-Carlo.

    :param data: List of data files.
    :type data: array
    :param wl_edges: Lower and upper bounds of the spectrum to load.
    :type wl_edges: 2-tuple
    :param null_key: Baseline to process.
    :type null_key: string
    :param nulls_to_invert: List of nulls to invert because their null and\
        antinull outputs are swapped.
    :type nulls_to_invert: list
    :param *args: Use dark data to get the error on the null depth.
    :type *args: TYPE
    :param **kwargs: Performs temporal binning of frames.
    :type **kwargs: TYPE
    :return:  Includes data to use for the fit: flux in (anti-)null and\
        photometric outputs, errors, wavelengths.
    :rtype: dict

    """
    '''
    Null table for getting the null and associated photometries 
    in the intermediate data
    Structure = Chosen null:[number of null, photometry A and photometry B]
    '''
    null_table = {'null1': [1, 1, 2], 'null2': [2, 2, 3], 'null3': [3, 1, 4],
                  'null4': [4, 3, 4], 'null5': [5, 3, 1], 'null6': [6, 4, 2]}

    indexes = null_table[null_key]

    null_data = []
    Iminus_data = []
    Iplus_data = []
    photo_data = [[], []]
    photo_err_data = [[], []]
    wl_scale = []
    
    lbti_mode = False
    piston_rms = []
    
    if 'lbti' in kwargs:
        if kwargs['lbti'] == True:
            lbti_mode = True

    for d in data:
        print(d)
        with h5py.File(d, 'r') as data_file:
            try:
                mask = np.where(np.array(data_file['nb_frames_null%s'%(indexes[0])]) >=100)[0]
                # mask = np.arange(np.array(data_file['Iminus%s' % (indexes[0])]).shape[0])
            except KeyError:
                mask = np.arange(np.array(data_file['Iminus%s' % (indexes[0])]).shape[0])

            wl_scale.append(np.array(data_file['wl_scale']))

#            null_data.append(np.array(data_file['null%s'%(indexes[0])]))
            Iminus_data.append(np.array(data_file['Iminus%s' % (indexes[0])])[mask])
            Iplus_data.append(np.array(data_file['Iplus%s' % (indexes[0])])[mask])

            # Fill with beam A intensity
            photo_data[0].append(np.array(data_file['p%s' % (indexes[1])])[mask])
            # Fill with beam B intensity
            photo_data[1].append(np.array(data_file['p%s' % (indexes[2])])[mask])
            # Fill with beam A error
            photo_err_data[0].append(
                np.array(data_file['p%serr' % (indexes[1])])[mask])
            # Fill with beam B error
            photo_err_data[1].append(
                np.array(data_file['p%serr' % (indexes[2])])[mask])

            if lbti_mode == True:
                # Fill with the piston rms
                piston_rms.append(np.array(data_file['piston_rms'])[mask])
            
            if 'null%s' % (indexes[0]) in nulls_to_invert:
                n = np.array(data_file['Iplus%s' % (indexes[0])])[mask] / \
                    np.array(data_file['Iminus%s' % (indexes[0])])[mask]
            else:
                n = np.array(data_file['Iminus%s' % (indexes[0])])[mask] / \
                    np.array(data_file['Iplus%s' % (indexes[0])])[mask]
            null_data.append(n)

    # Merge data along frame axis
    null_data = [selt for elt in null_data for selt in elt]
    Iminus_data = [selt for elt in Iminus_data for selt in elt]
    Iplus_data = [selt for elt in Iplus_data for selt in elt]
    
    if lbti_mode == True:
        piston_rms = [selt for elt in piston_rms for selt in elt]

    for i in range(2):
        photo_data[i] = [selt for elt in photo_data[i] for selt in elt]
        photo_err_data[i] = [selt for elt in photo_err_data[i] for selt in elt]


    null_data = np.array(null_data)
    Iminus_data = np.array(Iminus_data)
    Iplus_data = np.array(Iplus_data)
    photo_data = np.array(photo_data)
    photo_err_data = np.array(photo_err_data)
    if lbti_mode == True:
        piston_rms = np.array(piston_rms)

    '''
    All the wl scale are supposed to be the same, just pick up the first of
    the list
    '''

    wl_scale = wl_scale[0]
    mask = np.arange(wl_scale.size)

    wl_min, wl_max = wl_edges
    mask = mask[(wl_scale >= wl_min) & (wl_scale <= wl_max)]

    if 'flag' in kwargs:
        flags = kwargs['flag']
        mask = mask[flags]

    null_data = null_data[:, mask]
    Iminus_data = Iminus_data[:, mask]
    Iplus_data = Iplus_data[:, mask]
    photo_data = photo_data[:, :, mask]
    wl_scale = wl_scale[mask]
        
    null_data = np.transpose(null_data)
    photo_data = np.transpose(photo_data, axes=(0, 2, 1))
    Iminus_data = np.transpose(Iminus_data)
    Iplus_data = np.transpose(Iplus_data)

    if 'frame_binning' in kwargs:
        if not kwargs['frame_binning'] is None:
            if kwargs['frame_binning'] > 1:
                nb_frames_to_bin = int(kwargs['frame_binning'])
                null_data, dummy = binning(
                    null_data, nb_frames_to_bin, axis=1, avg=True)
                photo_data, dummy = binning(
                    photo_data, nb_frames_to_bin, axis=2, avg=True)
                Iminus_data, dummy = binning(
                    Iminus_data, nb_frames_to_bin, axis=1, avg=True)
                Iplus_data, dummy = binning(
                    Iplus_data, nb_frames_to_bin, axis=1, avg=True)

    out = {'null': null_data, 'photo': photo_data, 'wl_scale': wl_scale,
           'photo_err': photo_err_data, 'wl_idx': mask, 'Iminus': Iminus_data,
           'Iplus': Iplus_data}
    
    if lbti_mode == True:
        out['piston_rms'] = piston_rms

    if len(args) > 0:
        null_err_data = getErrorNull(out, args[0])
    else:
        null_err_data = np.zeros(null_data.shape)
    out['null_err'] = null_err_data

    return out


def getErrorNull(data_dic, dark_dic):
    """Compute the error of the null depth.

    :param data_dic: Dictionary of the data from ``load_data``.
    :type data_dic: dict
    :param dark_dic: Dictionary of the dark from ``load_data``.
    :type dark_dic: dict
    :return: Array of the error on the null depths.
    :rtype: array

    """
    var_Iminus = dark_dic['Iminus'].var(axis=-1)[:, None]
    var_Iplus = dark_dic['Iplus'].var(axis=-1)[:, None]
    Iminus = data_dic['Iminus']
    Iplus = data_dic['Iplus']
    null = data_dic['null']

    std_null = (null**2 * (var_Iminus/Iminus**2 + var_Iplus/Iplus**2))**0.5
    return std_null


def computeNullDepth(na, IA, IB, wavelength, opd, phase_bias, dphase_bias,
                     dark_null, dark_antinull, zeta_minus_A, zeta_minus_B,
                     zeta_plus_A, zeta_plus_B, spec_chan_width,
                     oversampling_switch, switch_invert_null):
    """Compute the null depth.

    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the ratio of the null over the antinull fluxes.

    :param na: Astrophysical null depth.
    :type na: float
    :param IA: Values of intensity of beam A in the fringe pattern.
    :type IA: array
    :param IB: Values of intensity of beam B in the fringe pattern.
    :type IB: array
    :param wavelength: Wavelength of the fringe pattern, in nm.
    :type wavelength: float
    :param opd: Value of OPD in nm.
    :type opd: array
    :param phase_bias: Achromatic phase offset in radian.
    :type phase_bias: float
    :param dphase_bias: Achromatic phase offset complement in radian\
        (originally supposed to be fitted but now set to 0).
    :type dphase_bias: float
    :param dark_null: Synthetic values of detector noise in the null output.
    :type dark_null: array
    :param dark_antinull: Synthetic values of detector noise in the\
        antinull output.
    :type dark_antinull: array
    :param zeta_minus_A: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_A: float
    :param zeta_minus_B: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_B: float
    :param zeta_plus_A: Value of the zeta coefficient between antinull and\
        photometric outputs for beam A.
    :type zeta_plus_A: float
    :param zeta_plus_B: Value of the zeta coefficient between antinull and\
        photometric outputs for beam B.
    :type zeta_plus_B: float
    :param spec_chan_width: Width of a spectral channel in nm.
    :type spec_chan_width: float
    :param oversampling_switch: If ``True``, the spectral channel is\
        oversampled and averaged to take into account the loss of\
        temporal coherence.
    :type oversampling_switch: bool
    :param switch_invert_null: If ``True``, the null and antinull sequences\
        are swapped because they are swapped on real data.
    :type switch_invert_null: bool
    :return: Synthetic sequence of null dephts, synthetic sequence of flux in\
        the null output, synthetic sequence of flux in the antinull output.
    :rtype: 3-tuple of arrays

    """
    visibility = (1 - na) / (1 + na)
    wave_number = 1./wavelength
    sine = cp.sin(2*np.pi*wave_number*(opd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(
            1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        sine = sine * sinc

    if switch_invert_null:  # Data was recorded with a constant pi shift
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A *
                                           zeta_minus_B) * visibility * sine  # + dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * \
            visibility * sine  # + dark_antinull
#        Iminus = cp.random.normal(Iminus, Iminus**0.5, size=Iminus.shape)
#        Iplus = cp.random.normal(Iplus, Iplus**0.5, size=Iplus.shape)
        Iminus = Iminus + dark_null
        Iplus = Iplus + dark_antinull
        null = Iplus / Iminus
    else:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A *
                                           zeta_minus_B) * visibility * sine  # + dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * \
            visibility * sine  # + dark_antinull
#        Iminus = cp.random.normal(Iminus, Iminus**0.5, size=Iplus.shape)
#        Iplus = cp.random.normal(Iplus, Iplus**0.5, size=Iplus.shape)
        Iminus = Iminus + dark_null
        Iplus = Iplus + dark_antinull
        null = Iminus / Iplus
    return null, Iminus, Iplus

def computeNullDepthLBTI(na, IA, IB, wavelength, opd, phase_bias, dphase_bias,
                     thermal_bckg, zeta_minus_A, zeta_minus_B,
                     zeta_plus_A, zeta_plus_B, spec_chan_width,
                     oversampling_switch, sigma_eps):
    """Compute the null depth.

    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the ratio of the null over the antinull fluxes.

    :param na: Astrophysical null depth.
    :type na: float
    :param IA: Values of intensity of beam A in the fringe pattern.
    :type IA: array
    :param IB: Values of intensity of beam B in the fringe pattern.
    :type IB: array
    :param wavelength: Wavelength of the fringe pattern, in nm.
    :type wavelength: float
    :param opd: Value of OPD in nm.
    :type opd: array
    :param phase_bias: Achromatic phase offset in radian.
    :type phase_bias: float
    :param dphase_bias: Achromatic phase offset complement in radian\
        (originally supposed to be fitted but now set to 0).
    :type dphase_bias: float
    :param thermal_bckg: Synthetic values of detector noise in the null output.
    :type thermal_bckg: array
    :param zeta_minus_A: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_A: float
    :param zeta_minus_B: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_B: float
    :param zeta_plus_A: Value of the zeta coefficient between antinull and\
        photometric outputs for beam A.
    :type zeta_plus_A: float
    :param zeta_plus_B: Value of the zeta coefficient between antinull and\
        photometric outputs for beam B.
    :type zeta_plus_B: float
    :param spec_chan_width: Width of a spectral channel in nm.
    :type spec_chan_width: float
    :param oversampling_switch: If ``True``, the spectral channel is\
        oversampled and averaged to take into account the loss of\
        temporal coherence.
    :type oversampling_switch: bool
    :param sigma_eps: rms of the phase residual on a single frame (fringe blurring).
    :type sigma_eps: float    
    :return: Synthetic sequence of null dephts, synthetic sequence of flux in\
        the null output, synthetic sequence of flux in the antinull output.
    :rtype: 3-tuple of arrays

    """
    visibility = (1 - na) / (1 + na)
    wave_number = 1./wavelength
    cosine = cp.cos(2*np.pi*wave_number*(opd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(
            1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        cosine = cosine * sinc

    blurring = (1 - 0.5*sigma_eps**2 + 0.125 * sigma_eps**4)
    Iminus = IA*zeta_minus_A + IB*zeta_minus_B + \
        2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A *
                                       zeta_minus_B) * visibility * \
            blurring * cosine
    Iplus = IA*zeta_plus_A + IB*zeta_plus_B + \
        2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B)
    Iminus = Iminus + thermal_bckg
    null = Iminus / Iplus

    return null, Iminus, Iplus

def computeNullDepthNoAntinull(IA, IB, wavelength, opd, dark_null, dark_antinull,
                               zeta_minus_A, zeta_minus_B, zeta_plus_A,
                               zeta_plus_B, spec_chan_width,
                               oversampling_switch, switch_invert_null):
    """Compute the null depth without antinull output.

    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the ratio of the null over the antinull fluxes.
    The antinull flux is considered as a pure constructive fringe.

    :param IA: Values of intensity of beam A in the fringe pattern.
    :type IA: array
    :param IB: Values of intensity of beam B in the fringe pattern.
    :type IB: array
    :param wavelength: Wavelength of the fringe pattern.
    :type wavelength: float
    :param opd: Value of OPD in nm.
    :type opd: array
    :param dark_null: Synthetic values of detector noise in the null output.
    :type dark_null: array
    :param dark_antinull: Synthetic values of detector noise in the antinull\
        output. 
    :type dark_antinull: array
    :param zeta_minus_A: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_A: float
    :param zeta_minus_B: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_B: float
    :param zeta_plus_A: Value of the zeta coefficient between antinull and\
        photometric outputs for beam A.
    :type zeta_plus_A: float
    :param zeta_plus_B: Value of the zeta coefficient between antinull and\
        photometric outputs for beam B.
    :type zeta_plus_B: float
    :param spec_chan_width: Width of a spectral channel in nm.
    :type spec_chan_width: float
    :param oversampling_switch: If ``True``, the spectral channel is\
        oversampled and averaged to take into account the loss of\
        temporal coherence.
    :type oversampling_switch: bool
    :param switch_invert_null: If ``True``, the null and antinull sequences\
        are swapped because they are swapped on real data.
    :type switch_invert_null: bool
    :return: synthetic sequence of flux in the null output,\
        synthetic sequence of flux in the antinull output.
    :rtype: 2-tuple of arrays

    """
    wave_number = 1./wavelength
    sine = cp.sin(2*np.pi*wave_number*(opd))
    if oversampling_switch:
        delta_wave_number = abs(
            1/(wavelength + spec_chan_width/2) - 1/(wavelength -
                                                    spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        sine = sine * sinc

    if switch_invert_null:  # Data was recorded with a constant pi shift
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B)
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * sine + \
            dark_antinull
    else:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) *\
            sine + dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B)
    return Iminus, Iplus


def computeNullDepthLinear(na, IA, IB, wavelength, opd, phase_bias,
                           dphase_bias, dark_null, dark_antinull,
                           zeta_minus_A, zeta_minus_B, zeta_plus_A,
                           zeta_plus_B, spec_chan_width, oversampling_switch,
                           switch_invert_null):
    """Compute the null depth from a linear expression.

    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the linear expression :math:`N =  N_a + N_{instr}`.

    :param na: Astrophysical null depth.
    :type na: float
    :param IA: Values of intensity of beam A in the fringe pattern.
    :type IA: array
    :param IB: Values of intensity of beam B in the fringe pattern.
    :type IB: array
    :param wavelength: Wavelength of the fringe pattern.
    :type wavelength: array
    :param opd:  Value of OPD in nm.
    :type opd: array
    :param phase_bias: Achromatic phase offset in radian.
    :type phase_bias: float
    :param dphase_bias: Achromatic phase offset complement in radian\
        (originally supposed to be fitted but now set to 0).
    :type dphase_bias: float
    :param dark_null: Synthetic values of detector noise in the null output.
    :type dark_null: array
    :param dark_antinull: Synthetic values of detector noise in the antinull\
        output. 
    :type dark_antinull: array
    :param zeta_minus_A: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_A: float
    :param zeta_minus_B: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_B: float
    :param zeta_plus_A: Value of the zeta coefficient between antinull and\
        photometric outputs for beam A.
    :type zeta_plus_A: float
    :param zeta_plus_B: Value of the zeta coefficient between antinull and\
        photometric outputs for beam B.
    :type zeta_plus_B: float
    :param spec_chan_width: Width of a spectral channel in nm.
    :type spec_chan_width: float
    :param oversampling_switch: If ``True``, the spectral channel is\
        oversampled and averaged to take into account the loss of\
            temporal coherence.
    :type oversampling_switch: bool
    :param switch_invert_null: If ``True``, the null and antinull sequences\
        are swapped because they are swapped on real data.
    :type switch_invert_null: bool
    :return: synthetic sequence of null depths.
    :rtype: array

    """
    astroNull = na
    wave_number = 1./wavelength
    sine = cp.sin(2*np.pi*wave_number*(opd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(
            1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        sine = sine * sinc

    if switch_invert_null:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B + 2 * \
            np.sqrt(IA * IB) * np.sqrt(zeta_minus_A *
                                       zeta_minus_B) * sine + dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B - 2 * \
            np.sqrt(IA * IB) * np.sqrt(zeta_plus_A *
                                       zeta_plus_B) * sine + dark_antinull
        null = Iplus / Iminus
    else:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B - 2 * \
            np.sqrt(IA * IB) * np.sqrt(zeta_minus_A *
                                       zeta_minus_B) * sine + dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B + 2 * \
            np.sqrt(IA * IB) * np.sqrt(zeta_plus_A *
                                       zeta_plus_B) * sine + dark_antinull
        null = Iminus / Iplus

    return null + astroNull, Iminus, Iplus


def computeHanot(na, IA, IB, wavelength, opd, phase_bias, dphase_bias,
                 dark_null, dark_antinull,
                 zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B,
                 spec_chan_width, oversampling_switch, switch_invert_null):
    """Compute the Hanot's null depth estimator.

    Compute the null depth from generated random values of photometries,\
        detector noise and OPD. 
    The estimator is the one used in Hanot et al. (2011)\
        (https://ui.adsabs.harvard.edu/abs/2011ApJ...729..110H/abstract).

    :param na: Astrophysical null depth.
    :type na: float
    :param IA: Values of intensity of beam A in the fringe pattern.
    :type IA: array
    :param IB: Values of intensity of beam B in the fringe pattern.
    :type IB: array
    :param wavelength: Wavelength of the fringe pattern.
    :type wavelength: float
    :param opd: Value of OPD in nm.
    :type opd: array
    :param phase_bias: Achromatic phase offset in radian.
    :type phase_bias: float
    :param dphase_bias: Achromatic phase offset complement in radian\
        (originally supposed to be fitted but now set to 0).
    :type dphase_bias: float
    :param dark_null: Synthetic values of detector noise in the null output.
    :type dark_null: array
    :param dark_antinull: Synthetic values of detector noise in the antinull\
        output.
    :type dark_antinull: array
    :param zeta_minus_A: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_A: float
    :param zeta_minus_B: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_B: float
    :param zeta_plus_A: Value of the zeta coefficient between antinull and\
        photometric outputs for beam A.
    :type zeta_plus_A: float
    :param zeta_plus_B: Value of the zeta coefficient between antinull and\
        photometric outputs for beam B.
    :type zeta_plus_B: float
    :param spec_chan_width: Width of a spectral channel in nm.
    :type spec_chan_width: float
    :param oversampling_switch: If ``True``, the spectral channel is\
        oversampled and averaged to take into account the loss of\
        temporal coherence.
    :type oversampling_switch: bool
    :param switch_invert_null: If ``True``, the null and antinull sequences\
        are swapped because they are swapped on real data.
    :type switch_invert_null: bool
    :return: synthetic sequence of null depths.
    :rtype: array

    """
    astroNull = na
    wave_number = 1./wavelength
    DeltaPhi = 2*np.pi*wave_number*(opd) + phase_bias + dphase_bias

    if switch_invert_null:
        dI = (IA*zeta_plus_A - IB*zeta_plus_B) / \
            (IA*zeta_plus_A + IB*zeta_plus_B)
        Nb = dark_antinull / (IA*zeta_plus_A + IB*zeta_plus_B)
    else:
        dI = (IA*zeta_minus_A - IB*zeta_minus_B) / \
            (IA*zeta_minus_A + IB*zeta_minus_B)
        Nb = dark_null / (IA*zeta_minus_A + IB*zeta_minus_B)

    null = 0.25 * (dI**2 + DeltaPhi**2)
    return null + astroNull + Nb


def computeNullDepthCos(na, IA, IB, wavelength, offset_opd, dopd, phase_bias,
                        dphase_bias, dark_null, dark_antinull,
                        zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B,
                        spec_chan_width, oversampling_switch):
    """Compute the null depth.

    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the ratio of the null over the antinull fluxes.
    The interferometric term uses a cosine and not a sine function.    

    :param na: Astrophysical null depth.
    :type na: float
    :param IA: Values of intensity of beam A in the fringe pattern.
    :type IA: array
    :param IB: Values of intensity of beam B in the fringe pattern.
    :type IB: array
    :param wavelength: Wavelength of the fringe pattern.
    :type wavelength: float
    :param opd: Value of OPD in nm.
    :type opd: array
    :param phase_bias: Achromatic phase offset in radian.
    :type phase_bias: float
    :param dphase_bias: Achromatic phase offset complement in radian\
        (originally supposed to be fitted but now set to 0).
    :type dphase_bias: float
    :param dark_null: Synthetic values of detector noise in the null output.
    :type dark_null: array
    :param dark_antinull: Synthetic values of detector noise in the antinull\
        output.
    :type dark_antinull: array
    :param zeta_minus_A: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_A: float
    :param zeta_minus_B: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_B: float
    :param zeta_plus_A: Value of the zeta coefficient between antinull and\
        photometric outputs for beam A.
    :type zeta_plus_A: float
    :param zeta_plus_B: Value of the zeta coefficient between antinull and\
        photometric outputs for beam B.
    :type zeta_plus_B: float
    :param spec_chan_width: Width of a spectral channel in nm.
    :type spec_chan_width: float
    :param oversampling_switch: If ``True``, the spectral channel is\
        oversampled and averaged to take into account the loss of\
        temporal coherence.
    :type oversampling_switch: bool
    :param switch_invert_null: If ``True``, the null and antinull sequences\
        are swapped because they are swapped on real data.
    :type switch_invert_null: bool
    :return: synthetic sequence of null depths.
    :rtype: array

    """
    visibility = (1 - na) / (1 + na)
    wave_number = 1./wavelength
    sine = cp.cos(2*np.pi*wave_number*(offset_opd + dopd) +
                  phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(
            1/(wavelength + spec_chan_width/2) - 1/(wavelength -
                                                    spec_chan_width/2))
        arg = np.pi*delta_wave_number * (offset_opd + dopd)
        sinc = cp.sin(arg) / arg
        sine = sine * sinc

    Iminus = IA*zeta_minus_A + IB*zeta_minus_B - \
        2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) *\
        visibility * sine + \
        dark_null
    Iplus = IA*zeta_plus_A + IB*zeta_plus_B + \
        2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) *\
        visibility * sine + \
        dark_antinull
    null = Iminus / Iplus
    return null


def computeNullDepthLinearCos(na, IA, IB, wavelength, opd, phase_bias,
                              dphase_bias, dark_null, dark_antinull,
                              zeta_minus_A, zeta_minus_B, zeta_plus_A,
                              zeta_plus_B, spec_chan_width,
                              oversampling_switch):
    """ Compute the null depth.

    Compute the null depth from generated random values of photometries,
    detector noise and OPD. 
    The estimator is the linear expression :math:`N =  N_a + N_{instr}`.
    The interferometric term uses a cosine and not a sine function.

    :param na: Astrophysical null depth.
    :type na: float
    :param IA: Values of intensity of beam A in the fringe pattern.
    :type IA: array
    :param IB: Values of intensity of beam B in the fringe pattern.
    :type IB: array
    :param wavelength: Wavelength of the fringe pattern.
    :type wavelength: float
    :param opd: Value of OPD in nm.
    :type opd: array
    :param phase_bias: Achromatic phase offset in radian.
    :type phase_bias: float
    :param dphase_bias: Achromatic phase offset complement in radian\
        (originally supposed to be fitted but now set to 0).
    :type dphase_bias: float
    :param dark_null: Synthetic values of detector noise in the null output.
    :type dark_null: array
    :param dark_antinull: Synthetic values of detector noise in the antinull\
        output.
    :type dark_antinull: array
    :param zeta_minus_A: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_A: float
    :param zeta_minus_B: Value of the zeta coefficient between null and\
        photometric outputs for beam B.
    :type zeta_minus_B: float
    :param zeta_plus_A: Value of the zeta coefficient between antinull and\
        photometric outputs for beam A.
    :type zeta_plus_A: float
    :param zeta_plus_B: Value of the zeta coefficient between antinull and\
        photometric outputs for beam B.
    :type zeta_plus_B: float
    :param spec_chan_width: Width of a spectral channel in nm.
    :type spec_chan_width: float
    :param oversampling_switch: If ``True``, the spectral channel is\
        oversampled and averaged to take into account the loss of\
        temporal coherence.
    :type oversampling_switch: bool
    :param switch_invert_null: If ``True``, the null and antinull sequences\
        are swapped because they are swapped on real data.
    :type switch_invert_null: bool
    :return: synthetic sequence of null depths.
    :rtype: array

    """
    astroNull = na
    wave_number = 1./wavelength
    cosine = cp.cos(2*np.pi*wave_number*(opd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(
            1/(wavelength + spec_chan_width/2) - 1/(wavelength -
                                                    spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        cosine = cosine * sinc

    Iminus = IA*zeta_minus_A + IB*zeta_minus_B - 2 * \
        np.sqrt(IA * IB) * np.sqrt(zeta_minus_A *
                                   zeta_minus_B) * cosine + dark_null
    Iplus = IA*zeta_plus_A + IB*zeta_plus_B + 2 * \
        np.sqrt(IA * IB) * np.sqrt(zeta_plus_A *
                                   zeta_plus_B) * cosine + dark_antinull
    null = Iminus / Iplus
    return null + astroNull


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


def getErrorCDF(data_null, data_null_err, null_axis):
    """Calculate the error of the CDF. It uses the cupy library.

    :param data_null: Null depth measurements used to create the CDF.
    :type data_null: array
    :param data_null_err: Error on the null depth measurements.
    :type data_null_err: array
    :param null_axis: Abscissa of the CDF.
    :type null_axis: array
    :return: Error of the CDF.
    :rtype: array

    """
    data_null = cp.asarray(data_null)
    data_null_err = cp.asarray(data_null_err)
    null_axis = cp.asarray(null_axis)
    var_null_cdf = cp.zeros(null_axis.size, dtype=cp.float32)
    for k in range(null_axis.size):
        prob = ndtr((null_axis[k]-data_null)/data_null_err)
        variance = cp.sum(prob * (1-prob), axis=-1)
        var_null_cdf[k] = variance / data_null.size**2

    std = cp.sqrt(var_null_cdf)
    return cp.asnumpy(std)


def getErrorPDF(data_null, data_null_err, null_axis):
    """Calculate the error of the PDF. It uses the cupy library.

    :param data_null: Null depth measurements used to create the PDF.
    :type data_null: array
    :param data_null_err: Error on the null depth measurements.
    :type data_null_err: array
    :param null_axis: Abscissa of the CDF.
    :type null_axis: array
    :return: Error of the PDF.
    :rtype: array

    """
    data_null = cp.asarray(data_null)
    data_null_err = cp.asarray(data_null_err)
    null_axis = cp.asarray(null_axis)
    var_null_hist = cp.zeros(null_axis.size-1, dtype=cp.float32)
    for k in range(null_axis.size-1):
        prob = ndtr((null_axis[k+1]-data_null)/data_null_err) - \
            ndtr((null_axis[k]-data_null)/data_null_err)
        variance = cp.sum(prob * (1-prob))
        var_null_hist[k] = variance / data_null.size**2

    std = cp.sqrt(var_null_hist)
    std[std == 0] = std[std != 0].min()
    return cp.asnumpy(std)


def doubleGaussCdf(x, mu1, mu2, sig1, sig2, A):
    """Calculate the CDF of the sum of two normal distributions.

    :param x: Abscissa of the CDF.
    :type x: array
    :param mu1: Location parameter of the first normal distribution.
    :type mu1: float
    :param mu2: Location parameter of the second normal distribution.
    :type mu2: float
    :param sig1: Scale parameter of the first normal distribution.
    :type sig1: float
    :param sig2: Scale parameter of the second normal distribution.
    :type sig2: float
    :param A: Relative amplitude of the second distribution with respect to the first one.
    :type A: float
    :return: CDF of the double normal distribution.
    :rtype: array

    """
    return sig1/(sig1+A*sig2) * ndtr((x-mu1)/(sig1)) + A*sig2/(sig1+A*sig2) * ndtr((x-mu2)/(sig2))


def getErrorBinomNorm(pdf, data_size, normed):
    """Calculate the error of the PDF knowing the number of elements in a bin\
        is a random value following a binomial distribution.

    :param pdf: Normalized PDF which the error is calculated.
    :type pdf: array
    :param data_size: Number of elements used to calculate the PDF.
    :type data_size: int
    :return: Error of the PDF.
    :rtype: array

    """
    if normed:
        pdf_err = ((pdf * (1 - pdf))/(data_size))**0.5  # binom-norm
    else:
        pdf_err = (pdf * (1 - pdf/data_size))**0.5  # binom-norm
    pdf_err[pdf_err == 0] = pdf_err[pdf_err != 0].min()
    return pdf_err


def rv_gen_doubleGauss(nsamp, mu1, mu2, sig, A, target):
    """Random values generator according to a double normal distribution with\
        the same scale factor.

    This function uses cupy to generate the values.

    :param nsamp: DESCRIPTION
    :type nsamp: int
    :param mu1: Location parameter of the first normal distribution.
    :type mu1: float
    :param mu2: Location parameter of the second normal distribution.
    :type mu2: float
    :param sig: Scale parameter of the both normal distributions.
    :type sig: float
    :param A: Relative amplitude of the second distribution with respect to\
        the first one.
    :type A: float
    :param target: If ``target = cpu``, the random values are transferred\
        from the graphic card memory to the RAM.
    :type target: str
    :return: Random values generated according to the double normal\
        distribution.
    :rtype: array or cupy array

    """
    x, step = cp.linspace(-2500, 2500, 10000, endpoint=False,
                          retstep=True, dtype=cp.float32)
    cdf = doubleGaussCdf(x, mu1, mu2, sig, A)
    cdf = cp.asarray(cdf, dtype=cp.float32)
    if target == 'cpu':
        rv = cp.asnumpy(rv_generator(x, cdf, nsamp))
    else:
        rv = rv_generator(x, cdf, nsamp)
        rv = cp.array(rv, dtype=cp.float32)
    return rv


def _wrap_func(func, xdata, ydata, transform):
    """Calculate the cost function to minimize

    Wrapper called by ``curvefit`` to calculate the cost function to minimize.

    Copy/pasted and adpated from
    https://github.com/scipy/scipy/blob/v1.5.4/scipy/optimize/minpack.py,
    line 481.

    :param func: function to fit.
    :type func: callable
    :param xdata: The independent variable where the data is measured.\
            Should usually be an M-length sequence or an (k,M)-shaped array\
            for functions with k predictors, but can actually be any object.
    :type xdata: array
    :param ydata: The dependent data, a length M array - nominally\
        ``f(xdata, ...)``.
    :type ydata: array
    :param transform: Weight on the data defined by :math:``1/\sigma``\
        where :math:``\sigma`` is the error on the y-values.

    :type transform: array
    :return: Cost function to minimize.
    :rtype: float

    """
    if transform is None:
        def func_wrapped(params):
            return func(xdata, *params) - ydata
    else:
        def func_wrapped(params):
            return transform * (func(xdata, *params) - ydata)

    return func_wrapped


def curvefit(func, xdata, ydata, p0=None, sigma=None, bounds=(None, None),
             diff_step=None, x_scale=1):
    """Fit the function of the NSC.

    Adaptation from the Scipy wrapper ``curve_fit``.
    The Scipy wrapper ``curve_fit`` does not give all the outputs of the
    least_squares function but gives the covariance matrix 
    (not given by the latter function).
    So I create a new wrapper giving both.
    I just copy/paste the code source of the official wrapper
    (https://github.com/scipy/scipy/blob/v1.5.4/scipy/optimize/minpack.py#L532-L834) 
    and create new outputs for getting the information I need.
    The algorithm is Trust-Reflective-Region.

    For exact documentation of the arguments ``diff_step`` and ``x_scale``,
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares.

    :param func: DESCRIPTION
    :type func: callable
    :param xdata: The independent variable where the data is measured.\
            Should usually be an M-length sequence or an (k,M)-shaped array\
            for functions with k predictors, but can actually be any object.
    :type xdata: array
    :param ydata: The dependent data, a length M array - nominally\
        ``f(xdata, ...)``.
    :type ydata: array
    :param p0: Initial guess for the parameters. If None, then the initial\
        values will all be 1 (if the number of parameters for the function\
                              can be determined using introspection,\
                                  otherwise a ValueError is raised),\
            defaults to None
    :type p0: list, optional
    :param sigma: Determines the uncertainty in `ydata`. If we define\
        residuals as ``r = ydata - f(xdata, *popt)``, defaults to None.
    :type sigma: None or M-length sequence, optional.
    :param bounds: Lower and upper bounds on parameters. Defaults\
        to no bounds. Each element of the tuple must be either an array with\
            the length equal to the number of parameters, or a scalar\
            (in which case the bound is taken to be the same for all\
             parameters). Use ``np.inf`` with an appropriate sign to disable\
            bounds on all or some parameters, defaults to (None, None).
    :type bounds: tuple of list-like, optional
    :param diff_step: Determines the relative step size for the finite\
        difference approximation of the Jacobian. The actual step is computed\
            as ``x * diff_step``. If None (default), then `diff_step` is\
            taken to be a conventional "optimal" power of machine epsilon for\
            the finite difference scheme used `William H. Press et. al.,\
                *“Numerical Recipes. The Art of Scientific Computing.\
                3rd edition”*, Sec. 5.7.`., defaults to None
    :type diff_step: None or scalar or array_like, optional
    :param x_scale: Characteristic scale of each variable. Setting `x_scale`\
        is equivalent to reformulating the problem in scaled variables\
            ``xs = x / x_scale``. An alternative view is that the size of a\
            trust region along jth dimension is proportional to\
            ``x_scale[j]``. Improved convergence may be achieved by setting\
            ``x_scale`` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to 'jac', the scale is iteratively updated using\
            the inverse norms of the columns of the Jacobian matrix\
            (as described in `J. J. More, “The Levenberg-Marquardt Algorithm:\
             Implementation and Theory,” Numerical Analysis, ed. G. A. Watson,\
            Lecture Notes in Mathematics 630, Springer Verlag, pp. 105-116,\
            1977.`), defaults to 1
    :type x_scale: array_like or scalar, optional
    :raises ValueError: "Unable to determine number of fit parameters".\
        If either `ydata` or `xdata` contain NaNs, or if incompatible options\
        are used.
    :return popt: Optimal values for the parameters so that the sum of the\
        squared residuals of ``f(xdata, *popt) - ydata`` is minimized.
    :rtype: array
    :return pcov: Optimal values for the parameters so that the sum of\
        the squared residuals of ``f(xdata, *popt) - ydata`` is minimized.
    :rtype: 2-D array
    :return res: `OptimizeResult` with the following fields defined:

            * x : ndarray, shape (n,)
                Solution found.
            * cost : float
                Value of the cost function at the solution.
            * fun : ndarray, shape (m,)
                Vector of residuals at the solution.
            * jac : ndarray, sparse matrix or LinearOperator, shape (m, n)
                Modified Jacobian matrix at the solution, in the sense that
                J^T J is a Gauss-Newton approximation of the Hessian of the
                cost function.
                The type is the same as the one used by the algorithm.
            * grad : ndarray, shape (m,)
                Gradient of the cost function at the solution.
            * optimality : float
                First-order optimality measure. In unconstrained problems,
                it is always the uniform norm of the gradient.
                In constrained problems, it is the quantity which was compared
                with `gtol` during iterations.
            * active_mask : ndarray of int, shape (n,)
                Each component shows whether a corresponding constraint is
                active
                (that is, whether a variable is at the bound):

                    *  0 : a constraint is not active.
                    * -1 : a lower bound is active.
                    *  1 : an upper bound is active.

                Might be somewhat arbitrary for 'trf' method as it generates\
                    a sequence of strictly feasible iterates and `active_mask`\
                    is determined within a tolerance threshold.
            * nfev : int
                Number of function evaluations done. Methods 'trf' and\
                    'dogbox' do not count function calls for numerical\
                    Jacobian approximation, as opposed to 'lm' method.
            * njev : int or None
                Number of Jacobian evaluations done. If numerical Jacobian
                approximation is used in 'lm' method, it is set to None.
            * status : int
                The reason for algorithm termination:

                    * -1 : improper input parameters status returned
                            from MINPACK.
                    *  0 : the maximum number of function evaluations is
                            exceeded.
                    *  1 : `gtol` termination condition is satisfied.
                    *  2 : `ftol` termination condition is satisfied.
                    *  3 : `xtol` termination condition is satisfied.
                    *  4 : Both `ftol` and `xtol` termination conditions are
                            satisfied.

            * message : str
                Verbal description of the termination reason.
            * success : bool
                True if one of the convergence criteria is satisfied
                (`status` > 0).
    :rtype: list
    """
    if bounds[0] is None:
        bounds[0] = -np.inf
    if bounds[1] is None:
        bounds[1] = np.inf

    if p0 is None:
        # determine number of parameters by inspecting the function
        from scipy._lib._util import getargspec_no_self as _getargspec
        args = _getargspec(func)[0]
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1
        p0 = np.ones(n,)
    else:
        p0 = np.atleast_1d(p0)

    if sigma is not None:
        sigma = np.array(sigma)
        transform = 1/sigma
    else:
        transform = None

    cost_func = _wrap_func(func, xdata, ydata, transform)
    
    jac = '3-point'
    res = least_squares(cost_func, p0, jac=jac, bounds=bounds, method='trf',
                        diff_step=diff_step, x_scale=x_scale, loss='huber',
                        verbose=2)  # , xtol=None)#, max_nfev=100)
    popt = res.x

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)

    warn_cov = False
    if pcov is None:
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warn_cov = True

    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning)

    return popt, pcov, res


def _objective_func(parameters, *args):
    func, xdata, ydata, transform = args
    if transform is None:
        obj_fun = func(xdata, *parameters) - ydata
    else:
        obj_fun = transform * (func(xdata, *parameters) - ydata)

    return np.sum(obj_fun**2)


# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    """Yield successive n-sized chunks from l.

    :param l: Size of the list to chunk.
    :type l: int
    :param n: Size of a chunk of the list.
    :type n: int
    :return: Generator of the chunks.
    :rtype: Yields

    """
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


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

    return ((fluxA, fluxB), (spectrumA, spectrumB))


def binning(arr, binning, axis=0, avg=False):
    """Bin elements together

    :param arr: Array containing data to bin
    :type arr: nd-array
    :param binning: Number of frames to bin
    :type binning: int
    :param axis: axis along which the frames are binned, defaults to 0
    :type axis: int, optional
    :param avg: if ``True``, the method returns the average of the
            binned frame. Otherwise, return its sum., defaults to False.
    :type avg: bool, optional
    :return: binned datacube, index of the kept frames.
    :rtype: 2-tuple

    """
    if binning is None or binning > arr.shape[axis]:
        binning = arr.shape[axis]

    shape = arr.shape
    # Number of frames which can be binned respect to the input value
    crop = shape[axis]//binning*binning
    arr = np.take(arr, np.arange(crop), axis=axis)
    shape = arr.shape
    if axis < 0:
        axis += arr.ndim
    shape = shape[:axis] + (-1, binning) + shape[axis+1:]
    arr = arr.reshape(shape)
    if not avg:
        arr = arr.sum(axis=axis+1)
    else:
        arr = arr.mean(axis=axis+1)

    cropped_idx = np.arange(crop).reshape(shape[axis], shape[axis+1])

    return arr, cropped_idx


def sortFrames(dic_data, nb_frames_to_bin, quantile, factor_minus, factor_plus,
               which_null, starname, plot=False, save_path=''):
    """Perform sigmal-clipping to remove frames non-normally distributed\
        phase fluctuations.

    Sigma-clipping to filter the frames which phase is not Gaussian
    (e.g. because of LWE).
    Fluxes of the null and antinull outputs are analysed in two steps.
    In the first step, for a given output, values between two thresholds are
    kept.
    The `base` is the upper bound for the antinull output or the lower bound
    for the null output.
    The base is defined as the median of the measurements which lower than
    the quantile (typically 10%) of the total sample in the null output,
    and upper for the antinull output.
    The second threshold is defined as the `base` plus or minus the standard
    deviation of the global sample wieghted by a coefficient.
    In the second step, frames for which both fluxes are kept are saved,
    the others are discarded.

    :param dic_data: Contains the extracted data from files by the function ``load_data``.
    :type dic_data: dict
    :param nb_frames_to_bin: Number of frames to bin before applying the filter.
        It is used to increase the SNR and exhibit the phase noise over
        the detector noise.
    :type nb_frames_to_bin: int
    :param quantile: Dirst quantile taken to determine the `base` threshold.
    :type quantile: loat between 0 and 1
    :param factor_minus: Factor applied to the std of the null flux to\
        determine the second threshold.
    :type factor_minus: float
    :param factor_plus: Factor applied to the std of the antinull flux
        to determine the second threshold.
    :type factor_plus: float
    :param which_null: Indicates on which baseline the filter is applied.
    :type which_null: string
    :param starname: Name of the star
    :type starname: string
    :param plot: If ``True``, it displays the time serie of the binned frames,
        the thresholds and highlights the filtered frames.
        The default is False., defaults to False
    :type plot: bool, optional
    :param save_path: Path where the plots is saved in png format (dpi = 150),
        defaults to ''
    :type save_path: str, optional
    :return new_dic: New dictionary with only the saved data points.
    :rtype: dict
    :return idx_good_frames: ndex of the kept frames in the input dictionary.
    :rtype: array

    """
    nb_frames_total = dic_data['Iminus'].shape[1]
    Iminus = dic_data['Iminus'].mean(axis=0)
    Iplus = dic_data['Iplus'].mean(axis=0)
    Iminus, cropped_idx_minus = binning(Iminus, nb_frames_to_bin, avg=True)
    Iplus, cropped_idx_plus = binning(Iplus, nb_frames_to_bin, avg=True)
    std_plus = Iplus.std()
    std_minus = Iminus.std()
#    std_plus = std_minus = max(std_plus, std_minus)

    x = np.arange(Iminus.size)
    Iminus_quantile = Iminus[Iminus <= np.quantile(Iminus, quantile)]
    Iminus_quantile_med = np.median(Iminus_quantile)
    Iplus_quantile = Iplus[Iplus >= np.quantile(Iplus, 1-quantile)]
    Iplus_quantile_med = np.median(Iplus_quantile)
    idx_plus = np.where(Iplus >= Iplus_quantile_med-factor_plus*std_plus)[0]
    idx_minus = np.where(Iminus <= Iminus_quantile_med +
                         factor_minus*std_minus)[0]
    idx_good_values = np.intersect1d(idx_plus, idx_minus)
    idx_good_frames = np.ravel(cropped_idx_plus[idx_good_values, :])

    new_dic = {}
    for key in dic_data.keys():
        new_dic[key] = dic_data[key]
        if dic_data[key].shape[-1] == nb_frames_total:
            new_dic[key] = np.take(new_dic[key], idx_good_frames, axis=-1)

    if plot:
        str_null = which_null.capitalize()
        str_null = str_null[:-1]+' '+str_null[-1]
        plt.figure(figsize=(19.2, 10.8))
        plt.title(str_null + ' %s %s' % (factor_minus, factor_plus), size=20)
        plt.plot(x, Iminus, '.', label='I-')
        plt.plot(x, Iplus, '.', label='I+')
        plt.plot(x, Iplus_quantile_med*np.ones_like(Iplus), 'r--', lw=3)
        plt.plot(x, (Iplus_quantile_med-factor_plus*std_plus)
                  * np.ones_like(Iplus), c='r', lw=3)
        plt.plot(x, Iminus_quantile_med*np.ones_like(Iminus), 'g--', lw=3)
        plt.plot(x, (Iminus_quantile_med+factor_minus*std_minus)
                  * np.ones_like(Iminus), c='g', lw=3)
        plt.plot(x[idx_good_values], Iminus[idx_good_values],
                  '+', label='Selected I-')
        plt.plot(x[idx_good_values], Iplus[idx_good_values],
                  'x', label='Selected I+')
        plt.grid()
        plt.legend(loc='best', fontsize=25)
        plt.xticks(size=25)
        plt.yticks(size=25)
        plt.ylabel('Intensity (count)', size=30)
        plt.xlabel('Frames', size=30)
        plt.tight_layout()
        string = starname + '_' + which_null + '_bin' +\
            str(nb_frames_to_bin) +\
            '_frame_selection_monitor_%s_%s' % (factor_minus, factor_plus)
        plt.savefig(save_path+string+'.png', dpi=150)

    return new_dic, idx_good_frames, (Iminus, Iplus)


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


def get_dark_cdf(dk, wl_scale0):
    """Get the CDF for generating RV from measured dark distributions.

    :param dk: dark data
    :type dk: array-like
    :param wl_scale0: wavelength axis
    :type wl_scale0: array
    :return: axis of the CDF and the CDF
    :rtype: 2-tuple
    """
    dark_size = [len(np.linspace(dk[i].min(), dk[i].max(),
                                 np.size(np.unique(dk[i])),
                                 endpoint=False))
                 for i in range(len(wl_scale0))]

    dark_axis = cp.array([np.linspace(dk[i].min(),
                                      dk[i].max(),
                                      min(dark_size), endpoint=False)
                          for i in range(len(wl_scale0))],
                         dtype=cp.float32)

    dark_cdf = cp.array([cp.asnumpy(computeCdf(dark_axis[i], dk[i],
                                               'cdf', True))
                         for i in range(len(wl_scale0))],
                        dtype=cp.float32)

    return dark_axis, dark_cdf


def update_label(old_label, exponent_text):
    """Format the label string with the exponent.

    :param old_label: label generated by matplotlib
    :type old_label: str
    :param exponent_text: exponent value
    :type exponent_text: str
    :return: new label with nice exponent format
    :rtype: str

    """
    if exponent_text == "":
        return old_label

    try:
        units = old_label[old_label.index(
            "(") + 1:old_label.rindex(")")]
    except ValueError:
        units = ""
    label = old_label.replace("({})".format(units), "")
    exponent_text = exponent_text.replace("\\times", "")

    if units != "":
        return "{} ({} {})".format(label, exponent_text, units)
    else:
        return "{} ({})".format(label, exponent_text)


def plot_photometries_histo(data_I, dk_photo, wl_scale, wl_idx0, nb_rows_plot,
                            count, photo_id, save_path,
                            activate_spectral_binning, skip_fit, key,
                            basin_hopping_count, wl_min, wl_max, datafolder):
    """Plot the histogram of the photometries per spectral channel

    :param data_I: array of the values of the photometry per spectral channel
    :type data_I: array
    :param dk_photo: array of the dark in the photometry per spectral channel
    :type dk_photo: array
    :param wl_scale: wavelength axis
    :type wl_scale: array
    :param wl_idx0: wavelength index
    :type wl_idx0: list
    :param nb_rows_plot: number of rows to display per figure
    :type nb_rows_plot: int
    :param count: counting gadget
    :type count: int
    :param photo_id: ID of the photometric tap
    :type photo_id: int or str
    :param save_path: path where to save the plot
    :type save_path: str

    """
    for wl_idx in wl_idx0:
        f = plt.figure(figsize=(19.20, 10.80))
        axs = []
        count = 0
        for wl in wl_idx:
            histo_I = np.histogram(data_I[wl], int(
                data_I[wl].size**0.5), density=True)            
            histo_dI = np.histogram(dk_photo[wl], int(
                np.size(dk_photo[wl])**0.5), density=True)
            
            if len(wl_idx) > 1:
                ax = f.add_subplot(nb_rows_plot, 2, count+1)
            else:
                ax = f.add_subplot(1, 1, count+1)
            axs.append(ax)
            plt.title('%.0f nm' % wl_scale[wl], size=20)
            plt.plot(histo_I[1][:-1], histo_I[0], '.',
                     markersize=5,
                     label='P%s' % (photo_id))
            plt.plot(histo_dI[1][:-1], histo_dI[0],
                     '.', markersize=5, label='Dark')            
            plt.grid()
            plt.legend(loc='best', fontsize=15)
            if list(wl_idx).index(wl) <= 1 or len(wl_idx) == 1:
                plt.xlabel('Flux (AU)', size=20)
            if count % 2 == 0:
                plt.ylabel('Frequency', size=20)
            plt.xticks(size=15)
            plt.yticks(size=20)
            count += 1
        plt.tight_layout()
        string = key + '_' + '%03d' % (basin_hopping_count) +\
            '_P%s' % (photo_id) + '_' +\
            str(wl_min) + '-' + str(wl_max) + '_' +\
            os.path.basename(datafolder[:-1]) +\
            '_%.0f' % (wl_scale[wl_idx[-1]])
        if not skip_fit:
            string = string + '_fit_pdf'
        if activate_spectral_binning:
            string = string + '_sb'
        plt.savefig(save_path+string+'.png', dpi=150)


def plot_chi2map(chi2map, mapx, mapy, mapz, argminz, stepx, stepy,
                 labelx, labely, labelz, key, save_path, x_id, y_id,
                 activate_spectral_binning, basin_hopping_count,
                 wl_min, wl_max, valmin, valmax):
    """Display the parameter space.

    :param chi2map: values of reduced Chi2 in the parameters space
    :type chi2map: array
    :param mapx: range of values of parameter on the x-axis
    :type mapx: array
    :param mapy: range of values of parameter on the y-axis
    :type mapy: array
    :param mapz: range of values of parameter on the z-axis
    :type mapz: array
    :param argminz: Index of the minimum Chi2 on the z-axis parameter
    :type argminz: int
    :param stepx: step in the range of values of parameter on the x-axis
    :type stepx: float
    :param stepy: step in range of values of parameter on the y-axis
    :type stepy: float
    :param labelx: Label of the x-axis
    :type labelx: str
    :param labely: Label of the y-axis
    :type labely: str
    :param labelz: Label of the z-axis (subplot title)
    :type labelz: str
    :param key: Considered baseline
    :type key: str
    :param save_path: path to save the plot
    :type save_path: str

    """
    save_name = save_path + '%s_%03d_chi2map_%s_vs_%s_%.0f-%.0f' % (
        key, basin_hopping_count, x_id, y_id, wl_min, wl_max)
    if activate_spectral_binning:
        save_name = save_name + '_sp'
    save_name = save_name + '.png'

    plt.figure(figsize=(19.20, 10.80))
    if mapz.size > 10:
        iteration = np.arange(argminz-5, argminz+5)
        if np.min(iteration) < 0:
            iteration -= iteration.min()
        elif np.max(iteration) > mapz.size - 1:
            iteration -= iteration.max() - mapz.size + 1
    else:
        iteration = np.arange(mapz.size)
    for i, it in zip(iteration, range(10)):
        plt.subplot(5, 2, it+1)
        plt.imshow(np.log10(chi2map[i]),
                   interpolation='none', origin='lower', aspect='auto',
                   extent=[mapx[0]-stepx/2,
                           mapx[-1]+stepx/2,
                           mapy[0]-stepy/2,
                           mapy[-1]+stepy/2])#,
                   # vmin=np.log10(valmin), vmax=np.log10(valmax))
        plt.colorbar()
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.title(labelz + ' %s' % mapz[i])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(key)
    plt.savefig(save_name, format='png', dpi=150)

def plot_lklh_map(lklhmap, mapx, mapy, mapz, argminz, stepx, stepy,
                 labelx, labely, labelz, key, save_path, x_id, y_id,
                 activate_spectral_binning, basin_hopping_count,
                 wl_min, wl_max, valmin, valmax):
    """Display the parameter space.

    :param chi2map: values of reduced Chi2 in the parameters space
    :type chi2map: array
    :param mapx: range of values of parameter on the x-axis
    :type mapx: array
    :param mapy: range of values of parameter on the y-axis
    :type mapy: array
    :param mapz: range of values of parameter on the z-axis
    :type mapz: array
    :param argminz: Index of the minimum Chi2 on the z-axis parameter
    :type argminz: int
    :param stepx: step in the range of values of parameter on the x-axis
    :type stepx: float
    :param stepy: step in range of values of parameter on the y-axis
    :type stepy: float
    :param labelx: Label of the x-axis
    :type labelx: str
    :param labely: Label of the y-axis
    :type labely: str
    :param labelz: Label of the z-axis (subplot title)
    :type labelz: str
    :param key: Considered baseline
    :type key: str
    :param save_path: path to save the plot
    :type save_path: str

    """
    save_name = save_path + '%s_%03d_lklh_map_%s_vs_%s_%.0f-%.0f' % (
        key, basin_hopping_count, x_id, y_id, wl_min, wl_max)
    if activate_spectral_binning:
        save_name = save_name + '_sp'
    save_name = save_name + '.png'

    plt.figure(figsize=(19.20, 10.80))
    if mapz.size > 10:
        iteration = np.arange(argminz-5, argminz+5)
        if np.min(iteration) < 0:
            iteration -= iteration.min()
        elif np.max(iteration) > mapz.size - 1:
            iteration -= iteration.max() - mapz.size + 1
    else:
        iteration = np.arange(mapz.size)
    for i, it in zip(iteration, range(10)):
        plt.subplot(5, 2, it+1)
        plt.imshow(np.log10(lklhmap[i]-np.min(lklhmap)),
                   interpolation='none', origin='lower', aspect='auto',
                   extent=[mapx[0]-stepx/2,
                           mapx[-1]+stepx/2,
                           mapy[0]-stepy/2,
                           mapy[-1]+stepy/2])#,
                   # vmin=np.log10(valmin), vmax=np.log10(valmax))
        plt.colorbar()
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.title('LKLH ' + labelz + ' %s' % mapz[i])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(key)
    plt.savefig(save_name, format='png', dpi=150)
    
def ramanujan(n):
    stirling = n * np.log(n) - n
    rama = stirling + 1/6 * np.log(8*n**3 + 4*n**2 + n + 1/30) + np.log(np.pi)/2
    try:
        rama[np.where(n==0)[0]] = 1
    except:
        pass
    return rama

def likelihood(params, data, func_model, *args, **kwargs):
    fact_n_i = ramanujan(data)
    x = args[0]
    
    if 'use_this_model' in kwargs.keys():
        model = kwargs['use_this_model']
    else:
        model = func_model(x, *params, args[1:], normed=False)
        
    if data.ndim == 2:
        model = model.reshape((data.shape[0], -1))

    logmodel = np.log(model)
    logsum = np.log(np.sum(model, 1, keepdims=True))
    logmodel -= logsum
    
    try:
        mini = np.min(logmodel[~np.isinf(logmodel)])
    except:
        mini = -15
    logmodel[np.isinf(logmodel)] = mini
    
    lklh = np.sum(data * logmodel - fact_n_i) #+ fact_n
 
    return -lklh
    
def likelihoodChi2(params, data, func_model, *args, **kwargs):
    """
    Maximization of the likelihood assuming the data points follow normal distributions.
    Hence we minimize the least squares which is a random variable following a Chi2 distribution.
    """
    xdata = args[0]
    if len(args) > 1 and args[1] is not None:
        data_err = args[1]
    else:
        data_err = np.ones_like(data)

    if 'use_this_model' in kwargs.keys():
        model = kwargs['use_this_model']
    else:
        model = func_model(xdata, *params, args[2:], normed=False)
        model = model.reshape((data.shape[0], -1))
        model = model / model.sum(1)[:,None] * data.sum(1)[:, None]
        model = model.ravel()
        
    chi2 = np.sum((data.ravel()- model)**2 / data_err.ravel()**2)
    chi2 = chi2 / (data.size - len(params)) # Get a reduced chi2
    
    return chi2
        
        
def optimize(func, minimizer, xdata, ydata, p0, yerr=None, bounds=None, diff_step=None):
    if bounds is not None:
        bounds_reformat = np.array(bounds)
        bounds_reformat = bounds_reformat.T
    res = minimize(minimizer, p0, args=(ydata, func, xdata, yerr), 
                   method='L-BFGS-B', jac='3-point',
                   bounds=bounds_reformat,
                   options={'finite_diff_rel_step':diff_step})
    popt = res.x
    pcov = res.hess_inv.todense()
    
    return popt, pcov, res
    

if __name__ == '__main__':
    n = int(1e3)
    n_samp_per_loop = int(1e6)
    nloop = 10
    rv = np.random.normal(0, 1, n)
    rv = rv.reshape((1, -1))
    wl_scale0 = np.array([11000.])
    histo1, bins1 = np.histogram(rv[0], int(rv.size**0.5))
    histo1 = histo1 / np.sum(histo1)
     
    rv_axis, rv_cdf = get_dark_cdf(rv, wl_scale0)
    
    # plt.figure()
    # plt.plot(cp.asnumpy(rv_axis[0]), cp.asnumpy(rv_cdf[0]), '.')
    # plt.grid()
    
    accum = []
    for k in range(nloop):
        rv2 = rv_generator(rv_axis[0], rv_cdf[0], n_samp_per_loop)
        rv2 = cp.asnumpy(rv2)
    
        try:
            histo2 = np.histogram(rv2, bins2)[0]
        except NameError:
            histo2, bins2 = np.histogram(rv2, int(rv2.size**0.5))
            
        histo2 = histo2 / np.sum(histo2)
        
        accum.append(histo2)
    
    accum = np.array(accum)
    accum = accum.sum(0)
    accum /= nloop
    
    plt.figure()
    plt.plot(bins1[:-1], histo1)
    plt.plot(bins2[:-1], histo2)
    plt.plot(bins2[:-1], accum)
    plt.grid()
    
    # rv = np.load('plop.npy')
    # rv_axis, rv_cdf = get_dark_cdf(rv, wl_scale0)
    
    # plt.figure()
    # plt.plot(cp.asnumpy(rv_axis[0]), cp.asnumpy(rv_cdf[0]), '.')
    # plt.grid()    

    # rv2 = rv_generator(rv_axis, rv_cdf, n_samp_per_loop)
    # rv2 = cp.asnumpy(rv2)
    
    # histo1 = np.histogram(rv[0], int(rv.size**0.5), density=True)
    # histo2 = np.histogram(rv2, int(rv2.size**0.5), density=True)
    
    # plt.figure()
    # plt.plot(histo1[1][:-1], histo1[0])
    # plt.plot(histo2[1][:-1], histo2[0])
    # plt.grid()
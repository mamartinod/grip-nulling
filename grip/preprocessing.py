#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module has preprocessing functions such as frame binning, frame sorting, list chunking.

Still work in progress.
"""

import numpy as np
import matplotlib.pyplot as plt

# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    """
    Yield successive n-sized chunks from l.

    Parameters
    ----------
    l : int
        Size of the list to chunk.
    n : int
        Size of a chunk of the list.

    Yields
    ------
    yield
        Generator of the chunks.

    """
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def binning(arr, binning, axis=0, avg=False):
    """
    Bin elements together.

    Parameters
    ----------
    arr : nd-array
        Array containing data to bin.
    binning : int
        Number of frames to bin.
    axis : int, optional
        Axis along which the frames are binned. The default is 0.
    avg : bool, optional
        if ``True``, the method returns the average of the\
                binned frame. Otherwise, return its sum. The default is False.

    Returns
    -------
    arr : nd-array
        Binned array.
    cropped_idx : array
        Index of the kept frames.

    """
    if binning is None or binning > arr.shape[axis] or binning < 0:
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
    """
    Perform sigmal-clipping to remove frames non-normally distributed\
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

    Parameters
    ----------
    dic_data : dict
        Contains the extracted data from files by the function ``load_data``.
    nb_frames_to_bin : int
        Number of frames to bin before applying the filter.\
            It is used to increase the SNR and exhibit the phase noise over\
            the detector noise.
    quantile : float between 0 and 1
        First quantile taken to determine the `base` threshold.
    factor_minus : float
        Factor applied to the std of the null flux to\
            determine the second threshold.
    factor_plus : float
        Factor applied to the std of the antinull flux\
            to determine the second threshold.
    which_null : string
        Indicates on which baseline the filter is applied.
    starname : string
        Name of the star.
    plot : bool, optional
        If ``True``, it displays the time serie of the binned frames,\
            the thresholds and highlights the filtered frames.\
            The default is False. The default is False.
    save_path : string, optional
        Path where the plots is saved in png format (dpi = 150). The default is ''.

    Returns
    -------
    new_dic : dict
       New dictionary with only the saved data points.
    idx_good_frames : array
        Index of the kept frames in the input dictionary.
    intensities: 2-tuple
        Arrays of fluxes in null and anti-null outputs.
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

    intensities = (Iminus, Iplus)
    return new_dic, idx_good_frames, intensities
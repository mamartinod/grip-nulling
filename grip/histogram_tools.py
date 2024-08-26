#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains function to build histograms, get cumulative density function
and generate randon values from customed distributions, both on GPU or CPU.
"""
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.special import ndtr, pdtr
    onGpu = True
except ModuleNotFoundError:
    import numpy as cp
    from scipy.special import ndtr
    onGpu = False
from functools import wraps

def counted_calls(f):
    @wraps(f)
    def count_wrapper(*args, **kwargs):
        count_wrapper.count += 1
        return f(*args, **kwargs)
    count_wrapper.count = 0
    return count_wrapper

       
@counted_calls
def create_histogram_model(params_to_fit, xbins, params_type, wl_scale0, instrument_model, instrument_args, rvus_forfit, cdfs, rvus, **kwargs):
    """
    .. _create-histogram-model:

    Monte-Carlo simulator of the instrument model to give a histogram.
    
    To avoid memory overflow, the total number of samples can be chunked into smaller parts.
    The resulting histogram is the same if the simulation is made with the total number of samples
    in one go.    

    Parameters
    ----------
    params_to_fit : tuple-like
        List of the parameters to fit.
    params_type : list
        Labels of the parameters to fit, see the notes for more information.
    xbins : 2D array
        1st axis = wavelength.
    wl_scale0 : 1D array
        Wavelength scale.
    instrument_model : function
        Function simulating the instrument.
    instrument_args : tuple, must contain same type of data (all float or all array of the same shape)
        List of arguments to pass to ``instrument_model`` which are not fitted.
    rvus_forfit : dic
        Contains the uniformly distributed values to generate random values following distributions which parameters are to fit.
    cdfs : tuple. First put CDF of quantities which does not depend on the wavelength.
        For wavelength-dependant quantity, 1st axis = wavelength.
    rvus : tuple. First put CDF of quantities which does not depend on the wavelength.
        For wavelength-dependant quantity, 1st axis = wavelength.
    \**kwargs : keywords
        ``n_samp_per_loop`` (int): number of samples for the MC simulation per loop.

        ``nloop`` (int): number of loops

    Returns
    -------
    accum : 1d-array
        Model of the histogram.
    diag : list
        Diagnostic data from the instrument model.
    diag_rv_1d : list
        Diagnostic data that are random values generated from the noise sources that are spectrally independant.
    diag_rv_2d : list
        Diagnostic data that are random values generated from the noise sources that are spectrally dependant.

    Notes
    -----
    This function can handle model with an arbitrary number of parameters, that can be parameters of a distribution or something else.
    
    The label system carried by `params_type` allows to identified the parameters that are not related to a distribution, 
    that are related to the same distribution (e.g. a Normal distribution needs 2 parameters, a Poisson needs one).
    
    In addition, when it comes to generate values from a distribution which parameters are to find, one may want some reproductibility.
    The parameter `rvus_forfit` embeds sequence of uniformly distributed values in a dictionary. 
    The keys of the dictionary must match the labels in `params_type`. If a key is not found, a new sequence is generated.
    For a key, the value can be `None` if no reproductibility is expected from this distribution.
    
    For example, let's assume a model that take the parameters: null depth, correcting factor, :math:`\mu` and :math:`\sigma` of 2 normal distributions
    and the :math:`\lambda` of a Poisson distribution.

    We have :
        - ``params_to_fit = [null depth, correcting factor, :math:`\mu_1` and :math:`\sigma_1`, :math:`\mu_1` and :math:\sigma_1`, :math:`\lambda`]``
        - ``params_type = ['deterministic', 'deterministic', 'normal1', 'normal1', 'normal2', 'normal2', 'poisson']``
        - ``rvus_forfit = {'normal1':None, 'normal2':array([0.27259743, 0.89770258, 0.72093494]), 'poisson':None}``
    
    The function will identified their are 2 *constant* parameters, 3 distributions to model with respectively 2, 2 and 1 parameters.
    Among these 3 distributions, only one is reproductible if given the same set of values between two calls of the function ``create_histogram_model``.
    
    Implemented distributions and their associated keywords are:
        - Normal distribution with keyword 'normal[...]' pattern, compatible with frozen uniform sequence
        - Poisson distribution with keyword 'poisson[...]' pattern, not compatible with frozen uniform sequence

    """

    dtypesize = cp.float32
    diag = []
    diag_rv_1d = []
    diag_rv_2d = []
    
    params_type = np.array(params_type)

    # """
    # Set some verbose to track the behaviour of the fitting algorithm
    # """
    count = create_histogram_model.count # Count number of times a function is called
    text_intput = (int(count), *params_to_fit)
    
    if 'verbose' in kwargs.keys() and kwargs['verbose'] == False:
        pass
    else:
        print(text_intput)

    # """
    # The user can choose to normalize the histogram by its sum or not
    # """
    if 'normed' in kwargs.keys() and kwargs['normed'] == False:
        normed = False
    else:
        normed = True



    # """
    # Set the parameters of the MC part of the creation of the histogram
    # """
    if 'n_samp_per_loop' in kwargs.keys():
        n_samp_per_loop = int(kwargs['n_samp_per_loop'])
    else:
        n_samp_per_loop = int(1e+7)
        
    if 'nloop' in kwargs.keys():
        nloop = kwargs['nloop']
    else:
        nloop = 1

    # """
    # Prepare the canvas to store the model of the histogram
    # """
    # Axes: spectral channel, number of bins
    accum = cp.zeros((xbins.shape[0], xbins.shape[1]-1), dtype=cp.float32)

    # """
    # Some random values do not depend on the spectral channels 
    # and must be generated out of the loop on the wavelength
    # """
    # Spot 1D & 2D cdfs
    cdfs_ndim = []
    for elt in cdfs:
        cdfs_ndim.append(elt[0].ndim)
    cdfs_ndim = np.array(cdfs_ndim)
    idx_1d_cdfs = np.where(cdfs_ndim == 1)[0] # e.g. injection or phase fluctuations
    idx_2d_cdfs = np.where(cdfs_ndim == 2)[0] # e.g. RON per spectral channel

    # """
    # Unpack the parameters to fit that are not parameters of statistical distributions.
    # """
    deterministic_params_to_fit = [elt for elt, elt2 in zip(params_to_fit, params_type) if elt2=='deterministic']

    # """
    # Generate random values from distributions which parameters are to be estimated.
    # """
    rvs_to_fit = generate_rv_params(params_to_fit, params_type, rvus_forfit, n_samp_per_loop, dtypesize)

    # """
    # Number of samples to simulate is high and the memory is low so we iterate
    # to create an average histogram
    # """
    for _ in range(nloop):
        diag_temp = []
        diag_rv_1d_temp = []
        diag_rv_2d_temp = []

        # """
        # Generation of the random values of the quantities independant from spectral channels
        # """
        rv1d_arr = cp.zeros((len(idx_1d_cdfs), n_samp_per_loop), dtype=dtypesize)
        # Generate random values from the 1D-cdfs
        for i in range(len(idx_1d_cdfs)):
            idx = idx_1d_cdfs[i]
            rv = rv_generator(cdfs[idx][0], cdfs[idx][1], n_samp_per_loop, rvus[idx])
            rv = rv.astype(dtypesize)
            rv1d_arr[i] = rv
        
        diag_rv_1d_temp.append(rv1d_arr)

        for k in range(wl_scale0.size):  # Iterate over the wavelength axis
            # """
            # Generate and pack random values of quantities which depend on the spectral axis.
            # """
            rv2d_arr = cp.zeros((len(idx_2d_cdfs), n_samp_per_loop), dtype=dtypesize)
            for i in range(len(idx_2d_cdfs)):
                idx = idx_2d_cdfs[i] # Select the quantity to MC
                rv = rv_generator(cdfs[idx][0][k], cdfs[idx][1][k], n_samp_per_loop, rvus[idx][k]) # Generate sequence of this quantity for the current spectral channel
                rv = rv.astype(dtypesize)
                rv2d_arr[i] = rv

            diag_rv_2d_temp.append(rv2d_arr)

            # """
            # Generate a signal delivered by the instrument given the input parameters
            # """
            out = instrument_model(deterministic_params_to_fit, rvs_to_fit, wl_scale0[k], k, *instrument_args, *rv1d_arr, *rv2d_arr)
            diag_temp.append(out[1:])
            out = out[0]

            # """
            # Clean the signal from NaNs.
            # """
            out = out[~cp.isnan(out)]  # Remove NaNs
            out = cp.sort(out)

            # """
            # Calculate the histogram of this signal
            # """
            bins = cp.asarray(xbins[k], dtype=cp.float32)
            pdf_null = cp.histogram(out, bins)[0]

            # """
            # Store it in the multi-spectral histogram
            # """
            if normed:
                accum[k] += pdf_null / cp.sum(pdf_null)
            else:
                accum[k] += pdf_null

            # """
            # End of loop on spectral channels
            # """

        diag.append(diag_temp)
        diag_rv_1d.append(diag_rv_1d_temp)
        diag_rv_2d.append(diag_rv_2d_temp)
        """
        End of loop on Monte-Carlo simulation
        """

    # Compute the average histogram over the nloops iterations
    accum = accum / nloop
    if cp.all(cp.isnan(accum)):
        accum[:] = 0

    if onGpu:
        accum = cp.asnumpy(accum)
        
    accum = accum.ravel()

    return accum, diag, diag_rv_1d, diag_rv_2d

def generate_rv_params(params_to_fit, params_type, rvus_forfit, n_samp, dtypesize):
    """
    Generate random values from distribution which parameters are to be fit.
    An arbitrary number of distributions can be used.
    
    This function returns random values from Normal and Poisson distributions.

    Parameters
    ----------
    params_to_fit : list
        List of all the parameters to fit.
    params_type : list
        List of the natures of the distribution from which generate random values. It must have the same length as `params_to_fit`
    rvus_forfit : dict
        Dictionary of sequence of uniform distributions that can be use to generate random values in a reproducible way. The keywords must be the same as the ones in `params_type`.
    n_samp : int
        Number of samples to generate per distribution.
    dtypesize : numpy or cupy digital size object
        Digital size of the array containing the random values (e.g. ``float``, ``int``, ``cp.float32``...).

    Raises
    ------
    NameError
        The distribution in `params_type` is not recognised.

    Returns
    -------
    rvs_to_fit : array
        Array containing all the random values from the distributions which parameters are to determine.
        A row contains the sequence from one distribution.
        
    See also
    --------
    :ref:`create_histogram_model <create-histogram-model>`: the Notes detail the use of ``params_to_fit`, ``params_type`` and ``rvus_forfit``.

    """
    params_to_fit = np.array(params_to_fit)
    dist_set = set(params_type)
    dist_set.remove('deterministic')
    dist_set = list(dist_set)
    rvs_to_fit = cp.zeros((len(dist_set), n_samp), dtype=dtypesize)
    
    for k in range(len(dist_set)):
        selected_idx = np.array([i for i,x in enumerate(params_type) if x==dist_set[k]])
        selected_dist = params_type[selected_idx]
        selected_params = params_to_fit[selected_idx]
        
        try:
            selected_rvu = rvus_forfit[dist_set[k]] # Read a dictionary
        except KeyError:
            selected_rvu = None
        
        if 'normal' in selected_dist[0]:
            mu, sig = selected_params
            axis = cp.linspace(mu - 6 * sig, mu + 6 * sig, 1001, dtype=dtypesize)
            cdf = ndtr((axis - mu) / sig)
            rv_forfit = rv_generator(axis, cdf, n_samp, selected_rvu)
            rv_forfit = rv_forfit.astype(dtypesize)

        elif 'poisson' in selected_dist[0]:
            lam = selected_params[0]
            rv_forfit = cp.random.poisson(lam, n_samp, dtype=dtypesize)
        else:
            raise NameError("Distribution not recognised. Only use among: 'normal', 'poisson'")
            
        rvs_to_fit[k] = rv_forfit
        
        return rvs_to_fit
    

def compute_data_histogram(data_null, bin_bounds, wl_scale, **kwargs):
    """
    Calculate the historam of the null depth for each spectral channel.
    By default, the histogram is normalised by its integral, \
        unless specified in ``**kwargs``.

    Parameters
    ----------
    data_null : 2d-array (wl, number of points)
        sequence of null depths. The first axis corresponds to the spectral \
            dispersion.
    bin_bounds : tuple-like (scalar, scalar)
        boundaries of the null depths range. Values out of this range are \
            pruned from ``data_null`` when making the histogram
    wl_scale : 1d-array
        wavelength scale.
    \**kwargs : extra-keywords
        Use ``normed=False`` to **not** normalise the histogram by its sum.

    Returns
    -------
    null_pdf : 2d-array (wavelength size, nb of bins)
        Histogram of the null depth per spectral channel.
    null_pdf_err : TYPE
        Error on the histogram frequency per spectral channel, assuming the \
            number of elements per bin follows a binomial distribution.

    """

    if 'nb_bins' in kwargs.keys():
        sz = kwargs['nb_bins'] + 1
    else:
        sz_list = np.array([np.size(d[(d >= bin_bounds[0]) &
                                      (d <= bin_bounds[1])])
                            for d in data_null])
        sz = np.max(sz_list)  # size of the sample of measured null depth
        sz = int(sz**0.5+1)

    null_axis = np.array([np.linspace(bin_bounds[0], bin_bounds[1], sz, retstep=False, dtype=np.float32)
        for i in range(data_null.shape[0])])

    #########
    # Compute the histogram per spectral channel and add it in the list
    # **null_pdf** and same for its error bars **null_pdf_err**.
    #########
    
    if 'normed' in kwargs.keys() and kwargs['normed'] == False:
        normed = False
    else:
        normed = True

    null_pdf = []
    null_pdf_err = []
    for wl in range(len(wl_scale)):
        pdf = np.histogram(data_null[wl], null_axis[wl], density=False)[0]
        pdf_size = np.sum(pdf)
        
        if normed:
            pdf = pdf / np.sum(pdf)
        else:
            pass
        null_pdf.append(pdf)
        pdf_err = getErrorBinomNorm(pdf, pdf_size, normed=normed)
        null_pdf_err.append(pdf_err)

    null_pdf = np.array(null_pdf)
    null_pdf_err = np.array(null_pdf_err)
    
    return null_axis, null_pdf, null_pdf_err, sz



if onGpu:
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
    
    binarySearchCuda = cp.ElementwiseKernel(
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

def computeCdf(x_axis, data, mode, normed):
    """
    Compute the empirical cumulative density function (CDF).
    It is a wrapper which calls the GPU or CPU version depending on the presence
    of cupy and a GPU.

    Parameters
    ----------
    x_axis : cupy array
        x-axis of the CDF.
    data : cupy array
        Data used to create the CDF.
    mode : string
        If ``ccdf``, the survival function (complementary of the CDF)\
            is calculated instead.
    normed : bool
        If ``True``, the CDF is normed so that the maximum is\
            equal to 1.

    Returns
    -------
    cdf : cupy array
        CDF of ``data``.

    """
    if onGpu:
        return computeCdfGPU(x_axis, data, mode, normed)
    else:
        return computeCdfCpu(x_axis, data, mode, normed)

def computeCdfGPU(x_axis, data, mode, normed):
    """
    Compute the empirical cumulative density function (CDF) on GPU with CUDA.

    Parameters
    ----------
    x_axis : cupy array
        x-axis of the CDF.
    data : cupy array
        Data used to create the CDF.
    mode : string
        If ``ccdf``, the survival function (complementary of the CDF)\
            is calculated instead.
    normed : bool
        If ``True``, the CDF is normed so that the maximum is\
            equal to 1.

    Returns
    -------
    cdf : cupy array
        CDF of ``data``.

    """
    cdf = cp.zeros(x_axis.shape, dtype=cp.float32)
    data = cp.asarray(data, dtype=cp.float32)
    absc = cp.asarray(x_axis, dtype=cp.float32)

    data = cp.sort(data)

    binarySearchCuda(absc, data, data.size, cdf)

    if mode == 'ccdf':
        cdf = data.size - cdf

    if normed:
        cdf = cdf/data.size

    return cdf

def binarySearchCpu(x, y):
    """
    Count values less than or equal to in another array.
    The algorithm used is binary search.
    
    
    https://www.enjoyalgorithms.com/blog/count-values-less-than-equal-to-in-another-array

    Parameters
    ----------
    x : 1d-array
        "reference" array.
    y : 1d-array
        Array in which we want to count the number of elements lower or equal to the values in `x`.
        MUST be sorted

    Returns
    -------
    high : int
        Number of elements less than or equal to in another array.

    """
    low = 0
    high = y.size
    mid = 0
    
    while low < high:
        mid = (low + high) // 2
        if y[mid] <= x:
            low = mid + 1
        else:
            high = mid
            
    return high
    
    
# def computeCdfCpu(x_axis, data, mode, normed):
#     """
#     Compute the empirical cumulative density function (CDF) on CPU.

#     Parameters
#     ----------
#     x_axis : cupy array
#         x-axis of the CDF.
#     data : cupy array
#         Data used to create the CDF.
#     mode : string
#         If ``ccdf``, the survival function (complementary of the CDF)\
#             is calculated instead.
#     normed : bool
#         If ``True``, the CDF is normed so that the maximum is\
#             equal to 1.

#     Returns
#     -------
#     cdf : cupy array
#         CDF of ``data``.

#     """

#     # First check if the data and the axies of the CDF are spectrally dispersed or not
#     if data.ndim == 1:
#         data = data.reshape((1, -1))
        
#     data = np.sort(data)
    
#     # Calculate the CDF by iterating over the spectral channel
#     cdfs = []
#     for k in range(data.shape[0]):
#         elt = data[k]
#         xelt = x_axis[k]
#         cdf = []
#         for x in xelt:
#             index = binarySearchCpu(x, elt)
#             cdf.append(index)
            
#         cdf = np.array(cdf)

#         if mode == 'ccdf':
#             cdf = data.size - cdf
        
#         if normed:
#             cdf = cdf / elt.size
            
#         cdfs.append(cdf)
        
#     cdfs = np.array(cdfs, dtype=np.float32)
#     return cdfs

def computeCdfCpu(x_axis, data, mode, normed):
    """
    Compute the empirical cumulative density function (CDF) on CPU.

    Parameters
    ----------
    x_axis : cupy array
        x-axis of the CDF.
    data : array
        Data used to create the CDF.
    mode : string
        If ``ccdf``, the survival function (complementary of the CDF)\
            is calculated instead.
    normed : bool
        If ``True``, the CDF is normed so that the maximum is\
            equal to 1.

    Returns
    -------
    cdf : cupy array
        CDF of ``data``.

    """
        
    data = np.sort(data)
    
    cdf = []
    for x in x_axis:
        index = binarySearchCpu(x, data)
        cdf.append(index)
        
    cdf = np.array(cdf)

    if mode == 'ccdf':
        cdf = data.size - cdf
    
    if normed:
        cdf = cdf / data.size
                
    cdf = np.array(cdf, dtype=np.float32)
    return cdf


def get_cdf(data):
    """
    Get the CDF of measured quantities.
    This function works on CPU and GPU.

    Parameters
    ----------
    data : array
        Data from which the CDF is wanted.
    wl_scale0 : array
        Wavelength axis.

    Returns
    -------
    axes : 2d-array
        axis of the CDF, first axis is the wavelength.
    cdfs : 2d-array
        CDF, first axis is the wavelength.

    """
    ndim0 = data.ndim 
    if ndim0 == 1:
        data = data.reshape((1, -1))

    sz = data.shape[0]
    sizes = [len(np.linspace(data[i].min(), data[i].max(),
                                 np.size(np.unique(data[i])),
                                 endpoint=True))
                 for i in range(sz)]

    axes = cp.array([np.linspace(data[i].min(),
                                      data[i].max(),
                                      min(sizes), endpoint=True)
                          for i in range(sz)],
                         dtype=cp.float32)

    cdfs = cp.array([computeCdf(axes[i], data[i], 'cdf', True) 
                     for i in range(sz)], dtype=cp.float32)


    if ndim0 == 1:
        axes = axes[0]
        cdfs = cdfs[0]

    return axes, cdfs


def get_dark_cdf(dk, wl_scale0):
    """
    Get the CDF for generating RV from measured dark distributions.

    Parameters
    ----------
    dk : array-like
        dark data.
    wl_scale0 : array
        wavelength axis.

    Returns
    -------
    dark_axis : nd-array
        axis of the CDF.
    dark_cdf : nd-array
        CDF.

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

    dark_cdf = cp.array([computeCdf(dark_axis[i], dk[i], 'cdf', True)
                          for i in range(len(wl_scale0))],
                        dtype=cp.float32)

    return dark_axis, dark_cdf


def rv_generator(absc, cdf, nsamp, rvu=None):
    """
    Random values generator based on the CDF.

    Parameters
    ----------
    absc : cupy array
        Abscissa of the CDF.
    cdf : cupy array
        Normalized arbitrary CDF to use to generate rv.
    nsamp : int
        Number of values to generate.
    rvu : TYPE, optional
        Use the same sequence of uniformly random values. The default is None.

    Returns
    -------
    output_samples : cupy array
        Sequence of random values following the CDF.

    """
    cdf, mask = cp.unique(cdf, True)
    cdf_absc = absc[mask]

    if rvu is None:
        try:
            rv_uniform = cp.random.rand(nsamp, dtype=cp.float32)
        except TypeError:
            rv_uniform = cp.random.rand(nsamp)
            rv_uniform = rv_uniform.astype(cp.float32)
    else:
        rv_uniform = cp.array(rvu, dtype=cp.float32)
        
    if onGpu:
        output_samples = cp.zeros(rv_uniform.shape, dtype=cp.float32)
        interpolate_kernel(rv_uniform, cdf, cdf.size, cdf_absc, output_samples)
    else:
        output_samples = np.interp(rv_uniform, cdf, cdf_absc, left=cdf_absc[0], right=cdf_absc[-1])

    return output_samples


def getErrorNull(data_dic, dark_dic):
    """
    Compute the error of the null depth.

    Parameters
    ----------
    data_dic : dict
        Dictionary of the data from ``load_data``.
    dark_dic : dict
        Dictionary of the dark from ``load_data``.

    Returns
    -------
    std_null : array
        Array of the error on the null depths.

    """
    var_Iminus = dark_dic['Iminus'].var(axis=-1)[:, None]
    var_Iplus = dark_dic['Iplus'].var(axis=-1)[:, None]
    Iminus = data_dic['Iminus']
    Iplus = data_dic['Iplus']
    null = data_dic['null']

    std_null = (null**2 * (var_Iminus/Iminus**2 + var_Iplus/Iplus**2))**0.5
    return std_null


def getErrorCDF(data_null, data_null_err, null_axis):
    """
    Calculate the error of the CDF. It uses the cupy library.

    Parameters
    ----------
    data_null : array
        Null depth measurements used to create the CDF.
    data_null_err : array
        Error on the null depth measurements.
    null_axis : array
        Abscissa of the CDF.

    Returns
    -------
    array
        Error of the CDF.

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
    """
    Calculate the error of the PDF. It uses the cupy library.

    Parameters
    ----------
    data_null : array
        Null depth measurements used to create the PDF.
    data_null_err : array
        Error on the null depth measurements.
    null_axis : array
        Abscissa of the CDF.

    Returns
    -------
    array
        Error of the PDF.

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



def getErrorBinomNorm(pdf, data_size, normed):
    """
    Calculate the error of the PDF knowing the number of elements in a bin\
        is a random value following a binomial distribution.

    Parameters
    ----------
    pdf : array
        Normalized PDF which the error is calculated.
    data_size : int
        Number of elements used to calculate the PDF.
    normed : bool
        Set to ``True`` if ``pdf`` is normalised, ``False`` otherwise.

    Returns
    -------
    pdf_err : array
        Error of the PDF.

    """
    if normed:
        pdf_err = ((pdf * (1 - pdf))/(data_size))**0.5  # binom-norm
    else:
        pdf_err = (pdf * (1 - pdf/data_size))**0.5  # binom-norm
    pdf_err[pdf_err == 0] = pdf_err[pdf_err != 0].min()
    return pdf_err

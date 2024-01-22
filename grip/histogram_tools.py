#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains function to build histograms, get cumulative density function
and generate randon values from customed distributions, both on GPU or CPU.
"""
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.special import ndtr
except ModuleNotFoundError:
    import numpy as cp
    from scipy.special import ndtr

class CountCalls:
    """
    This class counts the number of times a function is called.
    It is used as a decorator
    """
    def __init__(self, func): 
        self._count = 0 
        self._func = func 
    def __call__( self, *args, **kwargs): 
        self._count += 1 
        return self._func(*args,**kwargs) 
    @property 
    def call_count(self): 
        return self._count
    
    def reset_count(self):
        self._count = 0
    
@CountCalls
def create_histogram_model(params_to_fit, xbins, wl_scale0, instrument_model, instrument_args, rvu_forfit, cdfs, rvus, **kwargs):
    """
    Monte-Carlo simulator of the instrument model to give a histogram.
    
    To avoid memory overflow, the total number of samples can be chunked into smaller parts.
    The resulting histogram is the same if the simulation is made with the total number of samples
    in one go.

    Parameters
    ----------
    xbins : 2D array
        1st axis = wavelength.
    params_to_fit : tuple-like
        DESCRIPTION.
    wl_scale0 : 1D array
        DESCRIPTION.
    instrument_model : function
        DESCRIPTION.
    instrument_args : tuple, must contain same type of data (all float or all array of the same shape)
        DESCRIPTION.
    cdfs : tuple. First put CDF of quantities which does not depend on the wavelength.
        For wavelength-dependant quantity, 1st axis = wavelength.
    rvus : tuple. First put CDF of quantities which does not depend on the wavelength.
        For wavelength-dependant quantity, 1st axis = wavelength.
    **kwargs : keywords
        ``n_samp_per_loop`` (int): number of samples for the MC simulation per loop.\\
        ``nloop`` (int): number of loops

    Returns
    -------
    1d-array
        Model of the histogram.

    """
    dtypesize = cp.float32
    diag = []
    diag_rv_1d = []
    diag_rv_2d = []

    """
    Set some verbose to track the behaviour of the fitting algorithm
    """
    count = create_histogram_model.call_count # Count number of times a function is called
    text_intput = (int(count), *params_to_fit)
    
    if 'verbose' in kwargs.keys() and kwargs['verbose'] == False:
        pass
    else:
        print(text_intput)

    """
    The user can choose to normalize the histogram by its sum or not
    """
    if 'normed' in kwargs.keys() and kwargs['normed'] == False:
        normed = False
    else:
        normed = True

    """
    Unpack the parameters to fit.
    The astrophysical null or visibility MUST be the first argument.
    The two others MUST be the location and scale parameter of a normal distribution.
    """
    na = params_to_fit[0]
    mu = params_to_fit[1]
    sig = params_to_fit[2]

    """
    Set the parameters of the MC part of the creation of the histogram
    """
    if 'n_samp_per_loop' in kwargs.keys():
        n_samp_per_loop = int(kwargs['n_samp_per_loop'])
    else:
        n_samp_per_loop = int(1e+7)
        
    if 'nloop' in kwargs.keys():
        nloop = kwargs['nloop']
    else:
        nloop = 1

    """
    Prepare the canvas to store the model of the histogram
    """
    # Axes: spectral channel, number of bins
    accum = cp.zeros((xbins.shape[0], xbins.shape[1]-1), dtype=cp.float32)

    """
    Some random values do not depend on the spectral channels 
    and must be generated out of the loop on the wavelength
    """
    # Spot 1D & 2D cdfs
    cdfs_ndim = []
    for elt in cdfs:
        cdfs_ndim.append(elt[0].ndim)
    cdfs_ndim = np.array(cdfs_ndim)
    idx_1d_cdfs = np.where(cdfs_ndim == 1)[0] # e.g. injection or phase fluctuations
    idx_2d_cdfs = np.where(cdfs_ndim == 2)[0] # e.g. RON per spectral channel

    rv_forfit_axis = cp.linspace(mu - 6 * sig, mu + 6 * sig, 1001, dtype=dtypesize)
    rv_forfit_cdf = ndtr((rv_forfit_axis - mu) / sig)
    rv_forfit = rv_generator(rv_forfit_axis, rv_forfit_cdf, n_samp_per_loop, rvu_forfit)
    rv_forfit = rv_forfit.astype(dtypesize)

    """
    Number of samples to simulate is high and the memory is low so we iterate
    to create an average histogram
    """
    for _ in range(nloop):
        diag_temp = []
        diag_rv_1d_temp = []
        diag_rv_2d_temp = []

        """
        Generation of the random values of the quantities independant from spectral channels
        """
        rv1d_arr = cp.zeros((len(idx_1d_cdfs)+1, n_samp_per_loop), dtype=dtypesize)
        rv1d_arr[0] = rv_forfit
        # Generate random values from the 1D-cdfs
        for i in range(len(idx_1d_cdfs)):
            idx = idx_1d_cdfs[i]
            rv = rv_generator(cdfs[idx][0], cdfs[idx][1], n_samp_per_loop, rvus[idx])
            rv = rv.astype(dtypesize)
            rv1d_arr[i+1] = rv
        
        diag_rv_1d_temp.append(rv1d_arr)

        for k in range(wl_scale0.size):  # Iterate over the wavelength axis
            """
            Generate and pack random values of quantities which depend on the spectral axis.
            """
            rv2d_arr = cp.zeros((len(idx_2d_cdfs), n_samp_per_loop), dtype=dtypesize)
            for i in range(len(idx_2d_cdfs)):
                idx = idx_2d_cdfs[i] # Select the quantity to MC
                rv = rv_generator(cdfs[idx][0][k], cdfs[idx][1][k], n_samp_per_loop, rvus[idx][k]) # Generate sequence of this quantity for the current spectral channel
                rv = rv.astype(dtypesize)
                rv2d_arr[i] = rv

            diag_rv_2d_temp.append(rv2d_arr)

            """
            Generate a signal delivered by the instrument given the input parameters
            """
            out = instrument_model(na, wl_scale0[k], k, *instrument_args, *rv1d_arr, *rv2d_arr)
            diag_temp.append(out[1:])
            out = out[0]

            """
            Clean the signal from NaNs.
            """
            out = out[~cp.isnan(out)]  # Remove NaNs
            out = cp.sort(out)

            """
            Calculate the histogram of this signal
            """
            bins = cp.asarray(xbins[k], dtype=cp.float32)
            pdf_null = cp.histogram(out, bins)[0]

            """
            Store it in the multi-spectral histogram
            """
            if normed:
                accum[k] += pdf_null / cp.sum(pdf_null)
            else:
                accum[k] += pdf_null

            """
            End of loop on spectral channels
            """

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
    accum = cp.asnumpy(accum)

    return accum.ravel(), diag, diag_rv_1d, diag_rv_2d




def basin_hoppin_values(mu0, sig0, na0, bounds_mu, bounds_sig,
                        bounds_na):
    """Create several initial guesses.

    Create as many as initial guess as there are basin hopping iterations
    to do.
    The generation of these values are done with a normal distribution.

    :param mu0: Instrumental OPD around which random initial guesses are
                    created.
    :type mu0: float
    :param sig0: Instrumental OPD around which random initial guesses are
                    created.
    :type sig0: float
    :param na0: Null depth of the source around which random initial guesses
                are created.
    :type na0: float
    :param bounds_mu: Lower and upper bounds between which the random values of
                    **mu_opd** must be.
    :type bounds_mu: 2-tuple
    :param bounds_sig: Lower and upper bounds between which the random values
                    of **sig_opd** must be.
    :type bounds_sig: 2-tuple
    :param bounds_na: Lower and upper bounds between which the random values of
                    **na** must be.
    :type bounds_na: 2-tuple
    :return: New initial guess for **mu_opd**,
            new initial guess for **sig_opd**,
            new initial guess for **na**.
    :rtype: 3-tuple

    """
    print('Random withdrawing of init guesses')

    for _ in range(1000):
        mu_opd = np.random.normal(mu0, 50)
        if mu_opd > bounds_mu[0] and mu_opd < bounds_mu[1]:
            break
        if _ == 1000-1:
            print('mu_opd: no new guess, take initial one')
            mu_opd = mu0

    for _ in range(1000):
        sig_opd = abs(np.random.normal(sig0, 50.))
        if sig_opd > bounds_sig[0] and sig_opd < bounds_sig[1]:
            break
        if _ == 1000-1:
            print('sig opd: no new guess, take initial one')
            sig_opd = sig0

    for _ in range(1000):
        na = np.random.normal(na0, 0.03)
        if na > bounds_na[0] and na < bounds_na[1]:
            break
        if _ == 1000-1:
            print('na: no new guess, take initial one')
            na = na0

    print('Random drawing done')
#    np.random.set_state(orig_seed)
    return mu_opd, sig_opd, na




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
    **kwargs : extra-keywords
        Use ``normed=False`` to **not** normalise the histogram by its sum.

    Returns
    -------
    null_pdf : 2d-array (wavelength size, nb of bins)
        Histogram of the null depth per spectral channel.
    null_pdf_err : TYPE
        Error on the histogram frequency per spectral channel, assuming the \
            number of elements per bin follows a binomial distribution.

    """

    # sz = int(1e+6)  # size of the sample of measured null depth.
    sz_list = np.array([np.size(d[(d >= bin_bounds[0]) &
                                  (d <= bin_bounds[1])])
                        for d in data_null])
    sz = np.max(sz_list)  # size of the sample of measured null depth.

    null_axis = np.array([np.linspace(bin_bounds[0], bin_bounds[1], int(
        sz**0.5+1), retstep=False, dtype=np.float32)
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
    
    return null_axis, null_pdf, null_pdf_err, int(sz**0.5+1)




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

    

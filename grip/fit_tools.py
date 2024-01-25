#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the functions about fitting (chi2, likelihood, mcmc, exploring parameters).
"""
import numpy as np
from timeit import default_timer as time
from grip.histogram_tools import create_histogram_model
from itertools import product
import emcee
from scipy.optimize import OptimizeWarning, minimize, least_squares
from scipy.linalg import svd
import warnings

def explore_parameter_space(cost_fun, histo_data, param_bounds, param_sz, xbins, wl_scale0, instrument_model, instrument_args, rvu_forfit, cdfs, rvus, histo_err=None, **kwargs):
    """
    Explore the parameter space with a chosen optimizer (chi2, likelihood...)

    Parameters
    ----------
    cost_fun : function
        Cost function to use for model fitting.
    histo_data : nd-array
        Histograms of the data.
    param_bounds : nested tuple-like
        Nested tuple of the parameter bounds on the form ((min1, max1), (min2, max2)...).
    param_sz : list
        Number of points to sample each parameter axis.
    xbins : nd-array
        Bin axes of the histogram.
    wl_scale0 : 1d-array
        Wavelength scale.
    instrument_model : function
        Function simulating the instrument and noises.
    instrument_args : tuple
        Arguments for the ``instrument_model`` function.
    rvu_forfit : list of two arrays
        List of uniform random values use to generate normally distributed values with the fitting parameters ``mu`` and ``sig``.
    cdfs : list of arrays
        List of the CDF which are used to reproduce the statistics of the noises. There areas many sequences are noise sources to simulate.
    rvus : list of arrays
        List of uniform random values use to generate random values to reproduce the statistics of the noises. There areas many sequences are noise sources to simulate.
    histo_err : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : keywords
        Keywords to pass to the ``create_histogram_model`` function.

    Returns
    -------
    chi2map : nd-array
        Datacube containing the value of the cost function and the tested parameters.
    param_axes : TYPE
        DESCRIPTION.        
    steps : array
        Steps use to sample the parameters axes.

    """
    
    param_axes = [np.linspace(param_bounds[k][0], param_bounds[k][1], param_sz[k], endpoint=False, retstep=True)[0] for k in range(len(param_bounds))]
    steps = [np.linspace(param_bounds[k][0], param_bounds[k][1], param_sz[k], endpoint=False, retstep=True)[1] for k in range(len(param_bounds))]
    
    start = time()
    chi2map = []
    for param_values in product(*param_axes):
        parameters = np.array(param_values)
        out = create_histogram_model(parameters, xbins, wl_scale0, instrument_model, instrument_args, rvu_forfit, cdfs, rvus, **kwargs)[0]
        value = cost_fun(parameters, histo_data, create_histogram_model, histo_err, use_this_model=out)
        chi2map.append(value)
        
    chi2map = np.array(chi2map)
    chi2map = chi2map.reshape(param_sz)
    stop = time()
    print('Duration: %.3f s' % (stop-start))
    
    return chi2map, param_axes, steps

def ramanujan(n):
    """
    Ramanujan approximation to calculate the factorial of an integer. 
    Work very well for any integer >= 2.
    https://en.wikipedia.org/wiki/Stirling%27s_approximation

    Parameters
    ----------
    n : int or array
        Value to calculate its factorial.

    Returns
    -------
    rama : float or array
        Factorial of ``n``.

    """
    stirling = n * np.log(n) - n
    rama = stirling + 1/6 * np.log(8*n**3 + 4*n**2 + n + 1/30) + np.log(np.pi)/2
    try:
        rama[np.where(n==0)[0]] = 1
    except:
        pass
    return rama

def neg_log_multinomial(params, data, func_model, *args, **kwargs):
    """
    Likelihood of a dataset following multinomial distribution (e.g. number of occurences in the bins of a histogram)

    Parameters
    ----------
    params : array
        parameters to fit.
    data : array
        data to fit.
    func_model : function
        Model of the data.
    *args : function arguments
        extra-arguments to pass to this function and ``func_model``. The first argument must be values of the x-axis of the dataset. 
        If ``func_model`` takes any keyword, they must be passed in a dictionary in the last position in *args.
    **kwargs : keywords arguments
        Keywords accepted: 
            - `use_this_model` (array) : uses the values from a model generated out of this function instead of calling `func_model`
            - Keywords to pass to `func_model`

    Returns
    -------
    float
        negative log of the likelihood. The negative is picked for minimize algorithm to work.

    """
    fact_n_i = ramanujan(data)
    
    if 'use_this_model' in kwargs.keys():
        model = kwargs['use_this_model']
    else:
        if isinstance(args[-1], dict):
            kwargs = args[-1]
            args = list(args)
            args = args[:-1]
        model = func_model(params, *args, **kwargs)[0]
        
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
    
    lklh = np.sum(data * logmodel - fact_n_i)
 
    return -lklh
        
        
def minimize_fit(cost_func, func_model, p0, xdata, ydata, yerr=None, bounds=None, diff_step=None, func_args=(), func_kwargs={}):
    """
    Wrapper using the ``scipy.optimize.minimize`` with the L-BFGS-B algorithm.    

    Parameters
    ----------
    cost_func : function
        cost function.
    func_model : function
        model of the instrument.
    p0 : array-like
        Initial guess on the parameters to fit.
    xdata : array-like
        abscissa of the data to fit.
    ydata : array-like
        Dataset to fit (could be a histogram).
    yerr : array-like, optional
        uncertainties on the data. The default is None.
    bounds : TYPE, optional
        DESCRIPTION. The default is None.
    bounds : array-like, optional
        Boundaries of the parameters to fit. The shape must be like ((min_param1, max_param2), (min_param2, max_param2),...). The default is None.
    func_args : list-like, optional
        Arguments to pass to ``func_model``. The default is ().
    func_kwargs : dic-like, optional
        Keywords to pass to ``func_model``. The default is {}.

    Returns
    -------
    popt : array
        Best fitted values.
    pcov : 2D-array
        Covariance matrix.
    res : dic
        Complete return of the ``scipy.optimize.minimize`` function.

    """

    if yerr is None:
        temp = [ydata, func_model, xdata]
    else:
        temp = [ydata, func_model, yerr, xdata]

    func_args = list(func_args)
    func_args = temp + func_args

    if len(func_kwargs.keys()) > 0:
        func_args = func_args + [func_kwargs]
    
    func_args = tuple(func_args)
    res = minimize(cost_func, p0, args=func_args, 
                   method='L-BFGS-B', jac='3-point',
                   bounds=bounds,
                   options={'finite_diff_rel_step':diff_step})
    popt = res.x
    pcov = res.hess_inv.todense()
    
    return popt, pcov, res
    
def log_prior_uniform(params, bounds):
    """
    Uniform prior on a set of parameters to fit

    Parameters
    ----------
    params : array of size (N,)
        Parameters to fit.
    bounds : array-like
        Boundaries of the parameters to fit. The shape must be like ((min_param1, max_param2), (min_param2, max_param2),...).

    Returns
    -------
    float
        value of the prior.

    """
    return_log = 0
    for k in range(len(params)):
        if bounds[k][0] <= params[k] <= bounds[k][1]:
            return_log += 1
    if return_log == len(params):
        return 0.0
    else:
        return -np.inf


def log_posterior(params, lklh_func, bounds, func_model, data, func_args=(), func_kwargs={}, neg_lklh=True):
    """
    Posterior of the data.

    Parameters
    ----------
    params : list-like
        List of parameters.
    lklh_func : function
        Likelihood function to use.
    bounds : array-like
        Boundaries of the parameters to fit. The shape must be like ((min_param1, max_param2), (min_param2, max_param2),...).
    func_model : function
        Function of the model which reproduces the data to fig (e.g. histogram).
    data : array of size (N,) or (nb wl, N)
        Data to fit.
    func_args : list-like, optional
        Arguments to pass to ``func_model``. The default is ().
    func_kwargs : dic-like, optional
        Keywords to pass to ``func_model``. The default is {}.
    neg_lklh : bool, optional
        Change the sign of the value of the likelihood. 
        If ``True``, it means the returned likelihood by ``lklh_func`` is negative thus it signs must be reverted.
        The default is True.

    Returns
    -------
    log_posterior : float
        value of the posterior.

    """

    log_pr = log_prior_uniform(params, bounds)
    log_lklh = lklh_func(params, data, func_model, *func_args, **func_kwargs)
    
    if not neg_lklh:
        sign = 1.
    else:
        sign = -1.
        
    log_posterior = log_pr + sign * log_lklh
    
    if not np.isnan(log_posterior):
        return log_posterior
    else:
        return -np.inf


def mcmc(params, lklh_func, bounds, func_model, data, func_args=(), func_kwargs={}, 
             neg_lklh=True, nwalkers=6, nstep=2000, progress_bar=True):
    """
    Perform a MCMC with ``emcee`` library.

    Parameters
    ----------
    params : list-like
        Initial guess.
    lklh_func : callable function
        Function returning the likelihood.
    bounds : tuple-like
        Bounds of the parameters. The shape is ((min1, max1), ..., (minN, maxN)) .
    func_model : callable function
        Function of the model to fit.
    data : nd-array
        Data to fit.
    func_args : tuple, optional
        Tuple of arguments to pass to ``func_model``. The default is ().
    func_kwargs : dict, optional
        Dictionary of keywords to pass to ``func_model``. The default is {}.
    neg_lklh : bool, optional
        Indicates if the likelihood function returns a negative value. The default is True.
    nwalkers : int, optional
        Number of walkers to use in the MCMC. The default is 6.
    nstep : int, optional
        Number of steps for the walkers. The default is 2000.
    progress_bar : bool, optional
        Display the progress bar. The default is True.

    Returns
    -------
    samples : nd-array
        Samples from the MCMC algorithm. The shape is (``nwalkers``, ``nstep``)
    flat_samples : 1d-array
        Flatten chains with already discarded burn-in. \
            The burn-in values is defined as ``min(nstep//10, 600)``.

    """

    ndim = params.size
    norm_params = params.copy()
    pos = norm_params + 1e-7 * np.random.randn(nwalkers, params.size)
    
    posterior_func_args = (lklh_func, bounds, func_model, data, func_args, func_kwargs, neg_lklh)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=posterior_func_args)
    sampler.run_mcmc(pos, nstep, progress=progress_bar)
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=min(nstep//10, 600), flat=True)
    return samples, flat_samples


def calculate_chi2(params, data, func_model, *args, **kwargs):
    """
    Calculate a reduced Chi squared. It can be used by an optimizer.
    The Chi squared is calculated from the model function ``func_model`` or
    from a pre-calculated model (see Keywords).

    Parameters
    ----------
    params : array
        Guess of the parameters.
    data : nd-array
        Data to fit.
    func_model : callable function
        Model used to fit the data (e.g. model of the histogram).
    *args : list-like
        Extra-arguments which are in this order: the uncertainties (same shape as ``data``),\
            x-axis, arguments of ``func_model``.
    **kwargs : keywords
        Accepted keywords are: ``use_this_model`` to use a predefined model of the data;\
            keywords to pass to ``func_model``.

    Returns
    -------
    chi2 : float
        Reduced chi squared.

    """
    if len(args) >= 1 and args[0] is not None:
        data_err = args[0]
    else:
        data_err = np.ones_like(data)
        
    if 'use_this_model' in kwargs.keys():
        model = kwargs['use_this_model']
    else:
        model = func_model(params, *args[1:], **kwargs)[0]
        model = model.reshape((data.shape[0], -1))
        model = model / model.sum(1)[:,None] * data.sum(1)[:, None]
        model = model.ravel()
        
    chi2 = np.sum((data.ravel() - model)**2 / data_err.ravel()**2)
    red = data.size - len(params)
    chi2 = chi2 / red # Get a reduced chi2
    
    return chi2


def lstsqrs_fit(func_model, p0, xdata, ydata, yerr=None, bounds=None,
             diff_step=None, x_scale=1, func_args=(), func_kwargs={}):
    """
    Fit the function of the NSC.

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

    Parameters
    ----------
    func_model : function
        Function to fit the data.
    p0 : tuple-like
        Initial guess.
    xdata : 1d-array
        Flatten array of the x-axis of the data.
    ydata : 1d-array
        Flatten array of the data.
    yerr : 1d-array, optional
        Flatten array of the data error. The default is None.
    bounds : tuple-like, optional
        Tuple-like of shape ((min1, max1), ..., (minN, maxN)). The default is None.
    diff_step : list, optional
        Determines the relative step size for the finite difference approximation of the Jacobian.\
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares.\
                The default is None.
    x_scale : TYPE, optional
        Characteristic scale of each variable.\
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares. \
                The default is 1.
    func_args : tuple, optional
        Tuple of arguments to pass to the cost function. The default is ().
    func_kwargs : dict, optional
        Dictionary of arguments to pass to the cost function. The default is {}.

    Returns
    -------
    popt : list
        List of optimised parameters.
    pcov : 2d-array
        Covariance matrix of the optimised parameters.
    res : OptimizeResult
        Full output of the fitting algorithm.

    """
    
    bounds = np.array(bounds)
    
    if not bounds is None:
        bounds = np.array(bounds).T # Reformat in the correct format for least_squares

    p0 = np.atleast_1d(p0)

    cost_func = calculate_chi2
    func_args = list(func_args)
    func_args = [ydata, func_model, yerr, xdata] + func_args

    jac = '3-point'
    res = least_squares(cost_func, p0, jac=jac, bounds=bounds, method='trf',
                        diff_step=diff_step, x_scale=x_scale, loss='huber',
                        verbose=2, args=func_args, kwargs=func_kwargs)
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
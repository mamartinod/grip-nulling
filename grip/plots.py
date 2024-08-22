#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module gather plotting functions for GRIP tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from grip.preprocessing import divide_chunks

def plot_null_distributions(nb_rows_plot, wl_scale, text, null_axis, null_pdf,\
                            null_pdf_err, save_path, save_name, model=None,\
                                save_fig=True):
    """
    Plot the null distributions. There is one subplot per spectral channel.
    Each figure has a maximum number of subplots; several figures are generated
    if needed. Subplots layout is 2-column of a user-specified number of rows.
    The figures are saved in separate file, the name filed are index by the
    last wavelength displayed in the figure.

    Parameters
    ----------
    nb_rows_plot : int
        Number of rows in a figure instance.
    wl_scale : 1d-array
        Wavelength scale of the null depths.
    text : string
        Information to add on the plot, such as the values of the fitted parameters.
    null_axis : 2d-array (wavelength, nb of bins)
        x-axes of the histograms.
    null_pdf : 2d-array (wavelength, nb of bins)
        Histograms of the null depths.
    null_pdf_err : 2d-array (wavelength, nb of bins)
        Error bars on the histograms of the null depths.
    model : 1d- or 2d-array (wavelength, nb of bins), optional
        Model of the histograms. If ``None``, the model is not plotted. The default is None.
    save_path : string
        Path where to save.
    save_name : string
        Name under which the figures are saved.
    save_fig : bool, optional
        Save the figure or not. The default is True.

    Returns
    -------
    None.

    """
    
    nb_rows_plot = int(nb_rows_plot)

    wl_idx0 = np.arange(wl_scale.size)[::-1]
    # Subset of wavelength displayed in one figure
    wl_idx0 = list(divide_chunks(wl_idx0, nb_rows_plot*2))
    
    for wl_idx in wl_idx0:
        f = plt.figure(figsize=(19.20, 3.60*nb_rows_plot))
        count = 0
        axs = []
        for wl in wl_idx:
            if len(wl_idx) > 1:
                ax = f.add_subplot(nb_rows_plot, 2, count+1)
            else:
                ax = f.add_subplot(1, 1, count+1)
            axs.append(ax)
            plt.title('%.0f nm' % wl_scale[wl], size=30)
            plt.errorbar(
                null_axis[wl][:-1], null_pdf[wl],
                yerr=null_pdf_err[wl],
                fmt='.', markersize=10, label='Data')
            if not model is None:
                if model.ndim == 1:
                    out = model.reshape((wl_scale.size, -1))
                else:
                    out = model
                plt.errorbar(null_axis[wl][:-1],
                             out.reshape((wl_scale.size, -1))[wl],
                             markersize=10, lw=3, alpha=0.8,
                             label='Fit')
            plt.grid()
            if list(wl_idx).index(wl) >= len(wl_idx)-2 or\
                    len(wl_idx) == 1:
                plt.xlabel('Null depth', size=30)
            if count % 2 == 0:
                plt.ylabel('Frequency', size=30)
            plt.xticks(size=22)
            plt.yticks(size=22)
            exponent = np.floor(np.log10(null_pdf.max()))
            plt.ylim(-10**(exponent)/10, null_pdf.max()*1.05)
            # plt.xlim(-0.01, 0.5)
            count += 1
        plt.tight_layout(rect=[0., 0., 1, 0.88])
        #                plt.tight_layout()
        if len(wl_idx) > 1:
            axs[0].text(0.3, 1.52, text, va='center',
                        transform=axs[0].transAxes,
                        bbox=dict(boxstyle="square",
                                  facecolor='white'), fontsize=20)
        else:
            axs[0].text(0.25, 1.15, text, va='center',
                        transform=axs[0].transAxes,
                        bbox=dict(boxstyle="square",
                                  facecolor='white'), fontsize=17)

        wl_min = wl_scale[wl_idx[0]]
        wl_max = wl_scale[wl_idx[-1]]
        save_name2 = save_name + '_' + str(int(wl_min)) + '-' + str(int(wl_max))
        if save_fig:
            plt.savefig(save_path+save_name2+'.png', dpi=150)    
    
    
def plot_diag_spectral_data(nb_rows_plot, wl_scale, data_xy, labels, save_path, \
                    save_name, markers=['.', '-', '-'], save_fig=True):
    """
    Plot spectral data (one subplot per spectral channel).
    Each figure has a maximum number of subplots; several figures are generated
    if needed. Subplots layout is 2-column of a user-specified number of rows.
    The figures are saved in separate file, the name filed are index by the
    last wavelength displayed in the figure.    

    Different datasets can be put on the same subplot with the same or 
    different markers.

    Parameters
    ----------
    nb_rows_plot : int
        Number of rows in a figure instance.
    wl_scale : 1d-array
        Wavelength scale of the null depths.
    data_xy : nested tuples of shape (N, k, w, 2)
        Data to plot, N is the number of different markers, \
            k is the number of dataset to plot with the same marker, \
                w is te number of spectral channels (equal to the number of subplots),\
                    2 is the number of elements in the smallest tuple (x and y data).
    labels : nested tuples of shape (N, k)
        Labels for each plot.
    save_path : string
        Path where to save.
    save_name : string
        Name under which the figures are saved.
    markers : list of shape (N,), optional
        List of markers for all the dataset per subplot. The default is ['.', '-', '-'].
    save_fig : bool, optional
        Save the figure or not. The default is True.

    Raises
    ------
    TypeError
        Check if the sizes of ``data_xy``, ``markers`` and ``labels`` are consistent.

    Returns
    -------
    None.

    """
    
    if len(data_xy) != len(markers) and len(labels) != len(markers):
        raise TypeError("tuples 'data_xy' and 'labels' do not have the same size as 'markers'")
    elif len(data_xy) != len(markers):
        raise TypeError("tuples 'data_xy' does not have the same size as 'markers'")
    elif len(labels) != len(markers):
        raise TypeError("tuples 'labels' does not have the same size as 'markers'")


    nb_rows_plot = int(nb_rows_plot)

    wl_idx0 = np.arange(wl_scale.size)[::-1]
    # Subset of wavelength displayed in one figure
    wl_idx0 = list(divide_chunks(wl_idx0, nb_rows_plot*2))

    maxi = 0.
    
    for wl_idx in wl_idx0:
        f = plt.figure(figsize=(19.20, 10.80))
        count = 0
        axs = []
        for wl in wl_idx:
            if len(wl_idx) > 1:
                ax = f.add_subplot(nb_rows_plot, 2, count+1)
            else:
                ax = f.add_subplot(1, 1, count+1)
            axs.append(ax)
            plt.title('%.0f nm' % wl_scale[wl], size=20)
            
            for i in range(len(markers)):
                dat = data_xy[i]
                for k in range(len(dat)):
                    plt.plot(dat[k][wl][0], dat[k][wl][1], \
                             markers[i], label=labels[i][k])
                    # Let's find the maximum value ever plotted to adjust the scale
                    maxi = max(maxi, np.max(data_xy[i][k][wl][1])) 
            plt.grid()
            plt.xticks(size=15)
            plt.yticks(size=15)

            exponent = np.floor(np.log10(maxi))
            plt.ylim(-10**(exponent)/10, maxi*1.05)
            if count % 2 == 0:
                plt.ylabel('Frequency', size=20)
            if count == 0:
                plt.legend(loc='best', ncol=3, fontsize=15)
            if list(wl_idx).index(wl) <= 1 or len(wl_idx) == 1:
                plt.xlabel('Flux (AU)', size=20)
            count += 1
        plt.tight_layout()

        wl_min = wl_scale[wl_idx[0]]
        wl_max = wl_scale[wl_idx[-1]]
        save_name2 = save_name + '_' + str(int(wl_min)) + '-' + str(int(wl_max))
        if save_fig:
            plt.savefig(save_path+save_name2+'.png', dpi=150)  


def plot_diag_nonspectral_data(data_xy, labels, save_path, \
                    save_name, markers=['.', '-', '-'], save_fig=True):
    """
    Plot non spectral data.

    Different datasets can be put on the same subplot with the same or 
    different markers.    

    Parameters
    ----------
    data_xy : nested tuples of shape (N, k, w, 2)
        Data to plot, N is the number of different markers, \
            k is the number of dataset to plot with the same marker, \
                w is te number of spectral channels (equal to the number of subplots),\
                    2 is the number of elements in the smallest tuple (x and y data).
    labels : nested tuples of shape (N, k)
        Labels for each plot.
    save_path : string
        Path where to save.
    save_name : string
        Name under which the figures are saved.
    markers : list of shape (N,), optional
        List of markers for all the dataset per subplot. The default is ['.', '-', '-'].
    save_fig : bool, optional
        Save the figure or not. The default is True.

    Raises
    ------
    TypeError
        Check if the sizes of ``data_xy``, ``markers`` and ``labels`` are consistent.

    Returns
    -------
    None.

    """
    if len(data_xy) != len(markers) and len(labels) != len(markers):
        raise TypeError("tuples 'data_xy' and 'labels' do not have the same size as 'markers'")
    elif len(data_xy) != len(markers):
        raise TypeError("tuples 'data_xy' does not have the same size as 'markers'")
    elif len(labels) != len(markers):
        raise TypeError("tuples 'labels' does not have the same size as 'markers'")

    
    plt.figure(figsize=(19.20, 10.80))    
    for i in range(len(markers)):
        dat = data_xy[i]
        for k in range(len(dat)):
            plt.plot(data_xy[i][k][0], data_xy[i][k][1], \
                     markers[i], label=labels[i][k])
    plt.grid()
    plt.xticks(size=15)
    plt.yticks(size=15)

    plt.ylabel('Frequency', size=20)
    plt.legend(loc='best', ncol=3, fontsize=15)
    plt.xlabel('Flux (AU)', size=20)
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_path+save_name+'.png', dpi=150) 


def plot_parameter_space_2d(param_map, mapx, mapy, mapz, arg_axes, stepx, stepy,
                         labelx, labely, labelz, text, save_path,
                         x_id, y_id, basin_hopping_count,
                         wl_min, wl_max, save, valminmax=None):
    """
    Plot n-parameter spaces with heatmaps.
    It can only display along with the first 3 axes, all extra ones are sliced.
    The 3rd axis is displayed along several subplots of the two other axes.
    The maximum number of subplot is 10, and a single figure instance is displayed.
    The subplots display the +/- 5 values around the value indexed by ``argz``.
    If the z-axis has less than 10 points, they are all displayed in the instance.

    Parameters
    ----------
    param_map : nd-array
        Map of the parameter space (chi2, likelihood...).
    mapx : 1d-array
        Scale of the parameter to display along the x-axis of the heatmap. 
    mapy : 1d-array
        Scale of the parameter to display along the y-axis of the heatmap.
    mapz : 1d-array
        Scale of the parameter to display along the z-axis (subplot) of the heatmap..
    arg_axes : int or iterable
        Index of the remaining axes that are sliced. If it is an interger, 
        it is the extremum of the heat map to center the display of the z-axis, and the map must have 3 axes. 
        If it is an iterable, the first element must be the slice along the z-axis, ad the map must have 4 axes or more.
    stepx : float
        Step size of the x-axis scale.
    stepy : float
        Step size of the y-axis scale.
    labelx : string
        Label of the x-axis.
    labely : string
        Label of the y-axis.
    labelz : string
        Label of the title of the subplot.
    text : string
        Title of the plot and element in the naming of the saved file. Can be the name of the baseline for instance.
    save_path : string
        Path to the folder where the plot is saved.
    x_id : string
        Name of the x-axis.
    y_id : string
        Name of the y-axis.
    basin_hopping_count : int
        ID number of the basin hopping.
    wl_min : float
        Minimum wavelength value.
    wl_max : float
        Maximum wavelength value.
    save : bool
        Set to ``True`` to save the plot.
    valminmax : tuple-like, optional
        Set the minimum and maximum values of the dynamic range of the heatmap. The shape is (min value, max value). 
        If ``None``, the dynamics is set according to the minimum and maximum value of the whole array.
        The default is None.

    Returns
    -------
    None.
    
    Notes
    -----
    For heatmap of 2 axes, an artificial 3rd one must be created to use this function.

    """

    if valminmax is None:
        valmin = np.nanmin(param_map[~np.isinf(param_map)])
        valmax = np.nanmax(param_map[~np.isinf(param_map)])
    else:
        valmin, valmax = valminmax
        
    # The z-axis can be given in an iterable or as an integer
    try:
        argz = arg_axes[0]
    except TypeError:
        argz = arg_axes
    
    # If the map has more than 3 dimensions, we slice along the extra axes
    if param_map.ndim > 3:
        param_map = param_map[:,:,:,*arg_axes[1:]]


    plt.figure(figsize=(19.20, 10.80))
    if mapz.size > 10:
        iteration = np.arange(argz-5, argz+5)
        if np.min(iteration) < 0:
            iteration -= iteration.min()
        elif np.max(iteration) > mapz.size - 1:
            iteration -= iteration.max() - mapz.size + 1
    else:
        iteration = np.arange(mapz.size)


    for i, it in zip(iteration, range(10)):
        plt.subplot(5, 2, it+1)
        plt.imshow(param_map[i],
                   interpolation='none', origin='lower', aspect='auto',
                   extent=[mapx[0]-stepx/2,
                           mapx[-1]+stepx/2,
                           mapy[0]-stepy/2,
                           mapy[-1]+stepy/2],
                    vmin=valmin, vmax=valmax)
        plt.colorbar()
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.title(labelz + ' %s' % mapz[i])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(text)
    if save:
        save_name = save_path + '%s_%03d_param_map_%s_vs_%s_%.0f-%.0f' % (
            text, basin_hopping_count, x_id, y_id, wl_min, wl_max)
        save_name = save_name + '.png'
        plt.savefig(save_name, format='png', dpi=150)
        
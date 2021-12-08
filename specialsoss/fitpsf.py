# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction fitpsf method"""

from copy import copy
from bokeh.plotting import figure, show
from bokeh.models import Range1d
from bokeh.layouts import column
from hotsoss import locate_trace as lt
from hotsoss import plotting as plt
import numpy as np
import webbpsf

from . import utilities as u


def fit_2D_psf(psf):
    """
    Fit a 2D PSF of the given parameters to an image

    Parameters
    ----------
    frame

    Returns
    -------

    """
    # Get the NIRISS class from webbpsf and set the filter
    ns = webbpsf.NIRISS()
    ns.filter = 'CLEAR'
    ns.pupil_mask = 'GR700XD'
    psf = ns.calc_psf()[0].data.T
    psf += np.random.normal(loc=psf, scale=np.median(psf))
    dx, dy = psf.shape

    # # Fit the data using astropy.modeling
    # p_init = models.Polynomial2D(degree=2)
    # fit_p = fitting.LevMarLSQFitter()
    #
    # with warnings.catch_warnings():
    #     # Ignore model linearity warning from the fitter
    #     warnings.simplefilter('ignore')
    #     p = fit_p(p_init, x, y, z)
    #
    # # Plot the data with the best-fit model
    # plt.figure(figsize=(8, 2.5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(z, origin='lower', interpolation='nearest', vmin=-1e4, vmax=5e4)
    # plt.title("Data")
    # plt.subplot(1, 3, 2)
    # plt.imshow(p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
    #            vmax=5e4)
    # plt.title("Model")
    # plt.subplot(1, 3, 3)
    # plt.imshow(z - p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
    #            vmax=5e4)
    # plt.title("Residual")

    fig = figure()
    fig.image(image=[psf], dw=dx, dh=dy, x=0, y=0)
    show(fig)

    # # Fit the data using astropy.modeling
    # p_init = models.Polynomial2D(degree=2)
    # fit_p = fitting.LevMarLSQFitter()
    #
    # with warnings.catch_warnings():
    #     # Ignore model linearity warning from the fitter
    #     warnings.simplefilter('ignore')
    #     p = fit_p(p_init, x, y, z)
    #
    # # Plot the data with the best-fit model
    # plt.figure(figsize=(8, 2.5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(z, origin='lower', interpolation='nearest', vmin=-1e4, vmax=5e4)
    # plt.title("Data")
    # plt.subplot(1, 3, 2)
    # plt.imshow(p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
    #            vmax=5e4)
    # plt.title("Model")
    # plt.subplot(1, 3, 3)
    # plt.imshow(z - p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
    #            vmax=5e4)
    # plt.title("Residual")


def extract(data, filt='CLEAR', radius=25, subarray='SUBSTRIP256', contam_end=688, **kwargs):
    """
    Extract the time-series 1D spectra from a data cube

    Parameters
    ----------
    data: array-like (optional)
        The CLEAR+GR700XD or F277W+GR700XD datacube
    filt: str
        The name of the filter, ['CLEAR', 'F277W']
    pixel_masks: sequence
        The pixel masks for each order
    subarray: str
        The subarray name
    contam_end: idx
        The first column where the upper half of
        order 1 is not contaminated by order 2

    Returns
    -------
    dict
        The wavelength array and time-series 1D counts and spectra
    """
    # Dictionary for results
    results = {}

    # Load the wavebins
    order1_wave, order2_wave = lt.trace_wavelengths(order=None, wavecal_file=None, npix=10, subarray=subarray)
    order1_bins, order2_bins = lt.wavelength_bins(subarray=subarray)
    coeffs = lt.trace_polynomial(subarray=subarray)

    # Reshape
    data, dims = u.to_3d(data)

    # NaN reference pixels
    data = u.nan_reference_pixels(data)

    # ========================================================================================
    # Order 1 trace (lower half of trace is uncontaminated) ==================================
    # ========================================================================================

    # Load the pixel halfmasks for order 1
    order1_lowermask, order1_uppermask = halfmasks(radius=radius, subarray=subarray, order=1)

    # Get the counts in the trace halves
    order1_lowercount = u.bin_counts(data, order1_bins, pixel_mask=order1_lowermask)
    order1_uppercount = u.bin_counts(data, order1_bins, pixel_mask=order1_uppermask)

    # Add lowercount and uppercount in the areas of no contamination
    order1_counts_uncontam = order1_lowercount[:, contam_end:] + order1_uppercount[:, contam_end:]
    order1_unc_uncontam = np.ones_like(order1_counts_uncontam)

    # ...double lowercount in the areas of upper trace contamination
    order1_counts_contam = order1_lowercount[:, :contam_end] * 2
    order1_unc_contam = np.ones_like(order1_counts_contam)

    # Construct order 1 counts from contaminated and uncontaminated pieces
    order1_counts = np.concatenate([order1_counts_contam, order1_counts_uncontam], axis=1)
    order1_unc = np.concatenate([order1_unc_contam, order1_unc_uncontam], axis=1)

    # Plot for visual inspection
    order1_plot = plt.plot_frame(data[0] * order1_lowermask, cols=[10, 500, 1500], title='Order 1 Signal', trace_coeffs=coeffs)

    results['order1'] = {'counts': order1_counts, 'unc': order1_unc, 'wavelength': order1_wave, 'filter': filt, 'subarray': subarray, 'plot': order1_plot}

    # ========================================================================================
    # Order 2 trace (upper half of trace is uncontaminated) ==================================
    # ========================================================================================

    if filt == 'CLEAR':

        # Load the pixel halfmasks for order 1
        order2_lowermask, order2_uppermask = halfmasks(radius=radius, subarray=subarray, order=2)

        # Get the counts in the trace halves
        order2_lowercount = u.bin_counts(data, order2_bins, pixel_mask=order2_lowermask)
        order2_uppercount = u.bin_counts(data, order2_bins, pixel_mask=order2_uppermask)

        # Add lowercount and uppercount in the areas of no contamination
        order2_counts_uncontam = order2_lowercount[:, contam_end:] + order2_uppercount[:, contam_end:]
        order2_unc_uncontam = np.ones_like(order2_counts_uncontam)

        # ...double lowercount in the areas of upper trace contamination
        order2_counts_contam = order2_uppercount[:, :contam_end] * 2
        order2_unc_contam = np.ones_like(order2_counts_contam)

        # Construct order 1 counts from contaminated and uncontaminated pieces
        order2_counts = np.concatenate([order2_counts_contam, order2_counts_uncontam], axis=1)
        order2_unc = np.concatenate([order2_unc_contam, order2_unc_uncontam], axis=1)

        # Plot for visual inspection
        order2_plot = plt.plot_frame(data[0] * order2_lowermask, cols=[10, 500, 1500], title='Order 1 Signal', trace_coeffs=coeffs)

        results['order2'] = {'counts': order2_counts, 'unc': order2_unc, 'wavelength': order2_wave, 'filter': filt, 'subarray': subarray, 'plot': order2_plot}

    return results

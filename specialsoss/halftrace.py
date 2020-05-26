# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction halftrace method"""

import astropy.units as q
from bokeh.plotting import figure, show
from bokeh.models import Range1d
from bokeh.layouts import column
import numpy as np
from hotsoss import utils
from hotsoss import locate_trace as lt

from .utilities import combine_spectra, bin_counts, to_3d


def extract(data, filt, radius=25, subarray='SUBSTRIP256', contam_end=688, units=q.erg/q.s/q.cm**2/q.AA, **kwargs):
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
    units: astropy.units.quantity.Quantity
        The desired units for the output flux

    Returns
    -------
    dict
        The wavelength array and time-series 1D counts and spectra
    """
    # Dictionary for results
    results = {}

    # Load the wavebins
    wavelengths = lt.trace_wavelengths(order=None, wavecal_file=None, npix=10, subarray='SUBSTRIP256')
    wavebins = lt.wavelength_bins(subarray=subarray)

    # Reshape
    data, dims = to_3d(data)

    # Load the pixel halfmasks for order 1
    lowermask, uppermask = halfmasks(radius=radius, subarray=subarray)

    # Get the counts in the trace halves
    lowercount = bin_counts(data, wavebins[0], pixel_mask=lowermask)
    uppercount = bin_counts(data, wavebins[0], pixel_mask=uppermask)

    # Calculate a correction in the uncontaminated columns
    corrcount = (lowercount + uppercount) / (2. * lowercount)

    # fig1 = figure(width=1000)
    # fig1.line(range(1, 2047), uppercount[-1][1:-2], color='blue', legend='lower')
    # fig1.line(range(1, 2047), lowercount[-1][1:-2], color='red', legend='upper')
    #
    # fig2 = figure(width=1000)
    # for cc, color in zip(corrcount, ['blue','red','green','cyan']):
    #     fig2.line(range(2048), cc, color=color)
    #
    # show(column(fig1, fig2))
 
    return results


def halfmasks(subarray='SUBSTRIP256', radius=25, plot=False):
    """
    Generate a mask of the lower and upper halves of the first order trace

    Parameters
    ----------
    subarray: str
        The subarray to use, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
    radius: int
        The radius in pixels of the trace
    plot: bool
        Plot the masks

    Returns
    -------
    lowermask, uppermask
        The lower and upper masks for the array
    """
    # Get the first order trace
    trace = lt.trace_polynomial(order=1)
    trace_vals = lt.trace_polynomial(order=1, evaluate=True)
    centers = np.ceil(trace_vals).astype(int)

    # Split into lower and upper halves
    dims = 2048, 256
    lowermask = np.zeros(dims)
    uppermask = np.zeros_like(lowermask)
    for n, center in enumerate(centers):
        lowermask[n, center-radius:center] = 1
        uppermask[n, center+1:center+1+radius] = 1

    # The trace cuts across each center pixel at an angle so calculate the fraction based on the area.
    # Get average between pixel centers to calculate the trace value at each pixel bounds.
    avg = [trace_vals[0]]
    for n in range(1, 2048):
        avg.append((trace_vals[n-1]+trace_vals[n])/2.)
    avg.append(trace_vals[-1])

    # Calculate the fraction in the lower polygon
    rectangles = np.array([min(avg[n], avg[n+1])-np.floor(avg[n]) for n in range(2048)])
    triangles = np.array([0.5*(max(avg[n], avg[n+1])-min(avg[n], avg[n+1])) for n in range(2048)])
    lower_frac = rectangles + triangles

    # Make a mask for the upper and lower halves of the trace with 0 in masked pixels,
    # 1 in unmasked pixels, and the calculated fraction in the center pixel
    for n, (center, frac) in enumerate(zip(centers, lower_frac)):
        lowermask[n, center] = frac
        uppermask[n, center] = 1. - frac

    # Reshape into subarray
    if subarray == 'SUBSTRIP96':
        dims = 2048, 96
        lowermask = lowermask[:, :96]
        uppermask = uppermask[:, :96]

    if subarray == 'FULL':
        dims = 2048, 2048
        full = np.zeros(dims)
        lowermask = np.pad(lowermask, ((0, 0), (1792, 0)))
        uppermask = np.pad(uppermask, ((0, 0), (1792, 0)))

    # Plot it
    if plot:

        source = dict(lower=[lowermask.T], upper=[uppermask.T], total=[lowermask.T+uppermask.T])
        tooltips = [("(x,y)", "($x{int}, $y{int})"), ("Lower", "@lower"), ("Upper", "@upper"), ('Total', '@total')]
        mask_plot = figure(x_range=Range1d(0, 2048), y_range=Range1d(0, dims[1]), width=900, height=int(dims[1]), tooltips=tooltips)
        mask_plot.image(source=source, image='total', x=0, y=0, dh=dims[1], dw=2048, alpha=0.1)
        show(mask_plot)

    return lowermask.T, uppermask.T

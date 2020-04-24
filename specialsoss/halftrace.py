# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction halftrace method"""

import astropy.units as q
from bokeh.plotting import figure, show
from bokeh.models import Range1d
import numpy as np
from hotsoss import utils
from hotsoss import locate_trace as lt

from .utilities import combine_spectra, bin_counts


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

    return lowermask, uppermask


def extract(data, filt, pixel_masks=None, subarray='SUBSTRIP256', units=q.erg/q.s/q.cm**2/q.AA, **kwargs):
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

    # Get number of frames
    nframes = data.shape[0]

    # Load the pixel masks
    if pixel_masks is None:
        pixel_masks = np.ones((2, data.shape[-2], data.shape[-1]))

    # Get the mask for the upper and lower half of the trace

    # Extract the lower flux for all wavelengths

    # Extract the upper flux where uncontaminated

    # 

    # # Extract each order
    # for n, (wavelength, wavebin, mask) in enumerate(zip(wavelengths, wavebins, pixel_masks)):
    #
    #     # Bin the counts
    #     counts = bin_counts(data, wavebin, mask)
    #
    #     # Convert to flux
    #     flux = utils.counts_to_flux(wavelength, counts, filt=filt, subarray=subarray, order=n+1, units=units, **kwargs)
    #
    #     # TODO: Get the uncertainty
    #     unc = np.random.normal(loc=flux, scale=flux*0.01)
    #
    #     # Make arrays into monotonically increasing arrays
    #     idx = wavelength.argsort()
    #     wave = wavelength[idx]
    #     flux = flux[:, idx]
    #     unc = unc[:, idx]
    #     counts = counts[:, idx]
    #
    #     # Add result to the dictionary
    #     name = 'order{}'.format(n+1)
    #     results[name] = {'counts': counts, 'flux': flux, 'unc': unc, 'wavelength': wave, 'filter': filt, 'subarray': subarray}
    #
    # # Combine the order 1 and 2 spectra
    # combined = []
    # for n in range(nframes):
    #     spec1 = np.array([results['order1']['wavelength'], results['order1']['flux'][n], results['order1']['unc'][n]])
    #     spec2 = np.array([results['order2']['wavelength'], results['order2']['flux'][n], results['order2']['unc'][n]])
    #     spec3 = combine_spectra(spec1, spec2)
    #     combined.append(spec3)
    #
    # # Concatenate results and add to results
    # wave_final = combined[0][0]
    # flux_final = np.array([c[1] for c in combined])
    # unc_final = np.array([c[2] for c in combined])
    # results['final'] = {'wavelength': wave_final, 'counts': counts, 'flux': flux_final, 'unc': unc_final, 'filter': filt, 'subarray': subarray}
    #
    # return results

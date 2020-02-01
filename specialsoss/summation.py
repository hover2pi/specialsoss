# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction summation method"""

import astropy.units as q
from hotsoss import utils
import numpy as np


def extract(data, filt, subarray='SUBSTRIP256', units=q.erg/q.s/q.cm**2/q.AA, **kwargs):
    """
    Extract the time-series 1D spectra from a data cube

    Parameters
    ----------
    data: array-like
        The CLEAR+GR700XD or F277W+GR700XD datacube
    filt: str
        The name of the filter, ['CLEAR', 'F277W']
    subarray: str
        The subarray name
    units: astropy.units.quantity.Quantity
        The desired units for the output flux

    Returns
    -------
    dict
        The wavelength array and time-series 1D counts and spectra
    """
    # Make sure data is 3D
    if data.ndim == 4:
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))

    # Trim reference pixels
    # Left, right (all subarrays)
    data = data[:, :, 4:-4]

    # Top (excluding SUBSTRIP96)
    if subarray != 'SUBSTRIP96':
        data = data[:, :-4, :]

    # Bottom (Only FULL frame)
    if subarray == 'FULL':
        data = data[:, 4:, :]

    # Get total counts in each pixel column
    counts = np.nansum(data, axis=1)

    # Get the wavelength map
    wavemap = utils.wave_solutions(subarray=subarray, order=1, **kwargs)

    # Get mean wavelength in each column
    wavelength = np.nanmean(wavemap, axis=0)[4:-4]

    # Convert to flux using first order response
    flux = utils.counts_to_flux(wavelength, counts, filt=filt, subarray=subarray, order=1, units=units, **kwargs)

    # Make results dictionary
    results = {'final': {'wavelength': wavelength, 'counts': counts, 'flux': flux, 'filter': filt, 'subarray': subarray}}

    return results

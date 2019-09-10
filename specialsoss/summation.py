# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction summation method"""

from hotsoss import utils 
import numpy as np


def extract(data, wavecal, **kwargs):
    """
    Extract the time-series 1D spectra from a data cube

    Parameters
    ----------
    data: array-like
        The time-series 2D data
    wavemap: array-like
        The wavelength map for each order

    Returns
    -------
    dict
        The wavelength array and time-series 1D counts and spectra
    """
    # Calculate the mean wavelength for the first order
    wavelength = np.nanmean(wavecal[0], axis=0)

    # Get total counts in each pixel column
    counts = np.nansum(data, axis=2)

    # Convert to flux using first order response
    flux = utils.counts_to_flux(counts, order=1)

    return {'wavelength': wavelength, 'counts': counts, 'flux': flux}
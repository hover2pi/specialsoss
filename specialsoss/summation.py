# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction summation method"""

import astropy.units as q
from hotsoss import utils 
import numpy as np


def extract(data, filter='CLEAR', subarray='SUBSTRIP256', units=q.erg/q.s/q.cm**2/q.AA, **kwargs):
    """
    Extract the time-series 1D spectra from a data cube

    Parameters
    ----------
    data: array-like
        The time-series 2D data

    Returns
    -------
    dict
        The wavelength array and time-series 1D counts and spectra
    """
    # Get total counts in each pixel column
    counts = np.nansum(data, axis=2)

    # Get the wavelength
    wavemap = utils.wave_solutions(subarr=subarray, order=1, **kwargs)
    wavelength = np.nanmean(wavemap, axis=1)

    # Convert to flux using first order response
    flux = utils.counts_to_flux(wavelength, counts, filter=filter, subarray=subarray, order=1, units=units, **kwargs)

    return {'wavelength': wavelength, 'counts': counts, 'flux': flux}
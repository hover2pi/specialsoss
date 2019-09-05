# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction summation method"""

import numpy as np


def extract(data, wavemap, **kwargs):
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
    tuple
        The wavelength array and time-series 1D spectra
    """
    # Calculate the mean wavelength for the first order
    wavelength = np.nanmean(wavemap[0], axis=0)

    return wavelength, np.nansum(data, axis=2)
# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction summation method"""

import copy

from astropy.io import fits
import astropy.units as q
from bokeh.plotting import figure, show
import numpy as np
from pkg_resources import resource_filename

from . import locate_trace as lt


def extract(data):
    """
    Extract the time-series 1D spectra from a data cube

    Parameters
    ----------
    data: array-like
        The time-series 2D data

    Returns
    -------
    np.ndarray
        The time-series 1D spectra
    """
    return np.nansum(data, axis=1)
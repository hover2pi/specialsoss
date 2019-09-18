# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction summation method"""

import astropy.units as q
from hotsoss import utils 
import numpy as np


def extract(clear_data=None, f277w_data=None, subarray='SUBSTRIP256', units=q.erg/q.s/q.cm**2/q.AA, **kwargs):
    """
    Extract the time-series 1D spectra from a data cube

    Parameters
    ----------
    clear_data: array-like (optional)
        The CLEAR+GR700XD datacube
    f277w_data: array-like (optional)
        The F277W+GR700XD datacube
    filter: str
        The filter name, ['CLEAR', 'F277W']
    subarray: str
        The subarray name
    units: astropy.units.quantity.Quantity
        The desired units for the output flux

    Returns
    -------
    dict
        The wavelength array and time-series 1D counts and spectra
    """
    if clear_data is None and f277w_data is None:
        print("Please provide clear_data or f277w_data to perform extraction")
        return

    # Dictionary for results
    results = {}

    # Extract CLEAR + GR700XD
    for filt, data in zip(['CLEAR', 'F277W'], [clear_data, f277w_data]):
        if data is not None:

            # Make sure data is 3D
            if data.ndim == 4:
                data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))

            # Get total counts in each pixel column
            counts = np.nansum(data, axis=1)

            # Get the wavelength
            wavemap = utils.wave_solutions(subarray=subarray, order=1, **kwargs)
            wavelength = np.nanmean(wavemap, axis=0)

            # Convert to flux using first order response
            flux = utils.counts_to_flux(wavelength, counts, filt=filt, subarray=subarray, order=1, units=units, **kwargs)

            results[filt] = {'order1': {'wavelength': wavelength, 'counts': counts, 'flux': flux, 'filter': filt, 'subarray': subarray}}

    return results
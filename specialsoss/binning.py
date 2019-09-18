# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction binning method"""

import astropy.units as q
import numpy as np

from hotsoss import utils
from hotsoss import locate_trace as lt


def extract(clear_data=None, f277w_data=None, pixel_masks=None, subarray='SUBSTRIP256', units=q.erg/q.s/q.cm**2/q.AA, **kwargs):
    """
    Extract the time-series 1D spectra from a data cube

    Parameters
    ----------
    clear_data: array-like (optional)
        The CLEAR+GR700XD datacube
    f277w_data: array-like (optional)
        The F277W+GR700XD datacube
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
    if clear_data is None and f277w_data is None:
        print("Please provide clear_data or f277w_data to perform extraction")
        return

    # Dictionary for results
    results = {}

    # Load the wavebins
    wavelengths = lt.trace_wavelengths(order=None, wavecal_file=None, npix=10, subarray='SUBSTRIP256')
    wavebins = lt.wavelength_bins(subarray=subarray)

    # Extract the spectra for each filter and order
    for filt, data in zip(['CLEAR', 'F277W'], [clear_data, f277w_data]):
        if data is not None:

            # Make dictonary for the filter
            results[filt] = {}

            # Load the pixel masks
            if pixel_masks is None:
                pixel_masks = np.ones((2, data.shape[-2], data.shape[-1]))

            # Extract each order
            for n, (wavelength, wavebin, mask) in enumerate(zip(wavelengths, wavebins, pixel_masks)):

                # Results
                name = 'order{}'.format(n+1)
                result = {'wavelength': wavelength, 'filter': filt, 'subarray': subarray}

                # Bin the counts
                result['counts'] = bin_counts(data, wavebin, mask)

                # Convert to flux
                result['flux'] = utils.counts_to_flux(wavelength, result['counts'], filt=filt, subarray=subarray, order=n+1, units=units, **kwargs)

                # Add result to the dictionary
                results[filt][name] = result

    return results


def bin_counts(data, wavebins, pixel_mask=None):
    """
    Bin the counts in data given the wavelength bin information

    Parameters
    ----------
    data: array-like
        The 3D or 4D data to bin
    wavebins: sequence
        A list of lists of the pixels in each wavelength bin
    pixel_mask: array-like (optional)
        A 2D mask of 1s and 0s to apply to the data

    Returns
    -------
    np.ndarray
        The counts in each wavelength bin for each frame in data
    """
    # Reshape into 3D
    if data.ndim == 4:
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))

    # Array to store counts
    counts = np.zeros((data.shape[0], len(wavebins)))

    # Apply the pixel mask by multiplying non-signal pixels by 0 before adding
    if isinstance(pixel_mask, np.ndarray) and pixel_mask.shape == data.shape[1:]:
        data *= pixel_mask[None, :, :]

    # Add up the counts in each bin in each frame
    for n, (xpix, ypix) in enumerate(wavebins):
        counts[:, n] = np.nansum(data[:, xpix, ypix], axis=1)

    return counts

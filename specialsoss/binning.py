# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction binning method"""

import astropy.units as q
import numpy as np
from hotsoss import utils
from hotsoss import locate_trace as lt

from .utilities import combine_spectra


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

    # Load the pixel masks
    if pixel_masks is None:
        pixel_masks = np.ones((2, data.shape[-2], data.shape[-1]))

    # Extract each order
    for n, (wavelength, wavebin, mask) in enumerate(zip(wavelengths, wavebins, pixel_masks)):

        # Bin the counts
        counts = bin_counts(data, wavebin, mask)

        # Convert to flux
        flux = utils.counts_to_flux(wavelength, counts, filt=filt, subarray=subarray, order=n+1, units=units, **kwargs)

        # TODO: Get the uncertainty
        unc = np.random.normal(loc=flux*0.01)

        # Make into an array for easy combining
        spectrum = np.array([wavelength, flux, unc])

        # Make arrays into monotonically increasing arrays
        idx = spectrum[0].argsort()
        spectrum = spectrum[:, idx]
        counts = counts[idx]
        wavelength, flux, unc = spectrum

        # Add result to the dictionary
        name = 'order{}'.format(n+1)
        results[name] = {'counts': counts, 'flux': flux, 'unc': unc, 'wavelength': wavelength, 'filter': filt, 'subarray': subarray}

    # Combine the order 1 and 2 spectra
    combined = combine_spectra(results['order1']['spectrum'], results['order2']['spectrum'])
    results['final'] = {'flux': combined[1], 'wavelength': combined[0], 'unc': combined[2], 'filter': filt, 'subarray': subarray}

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
    counts = np.zeros((data.shape[0], len(wavebins)), dtype=float)

    # Apply the pixel mask by multiplying non-signal pixels by 0 before adding
    if isinstance(pixel_mask, np.ndarray) and pixel_mask.shape == data.shape[1:]:
        data *= pixel_mask[None, :, :]

    # Add up the counts in each bin in each frame
    for n, (xpix, ypix) in enumerate(wavebins):
        counts[:, n] = np.nansum(data[:, xpix, ypix], axis=1)

    return counts

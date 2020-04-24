# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction binning method"""

import astropy.units as q
import numpy as np
from hotsoss import utils
from hotsoss import locate_trace as lt

from .utilities import combine_spectra, bin_counts


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

    # Extract each order
    for n, (wavelength, wavebin, mask) in enumerate(zip(wavelengths, wavebins, pixel_masks)):

        # Bin the counts
        counts = bin_counts(data, wavebin, mask)

        # Convert to flux
        flux = utils.counts_to_flux(wavelength, counts, filt=filt, subarray=subarray, order=n+1, units=units, **kwargs)

        # TODO: Get the uncertainty
        unc = np.random.normal(loc=flux, scale=flux*0.01)

        # Make arrays into monotonically increasing arrays
        idx = wavelength.argsort()
        wave = wavelength[idx]
        flux = flux[:, idx]
        unc = unc[:, idx]
        counts = counts[:, idx]

        # Add result to the dictionary
        name = 'order{}'.format(n+1)
        results[name] = {'counts': counts, 'flux': flux, 'unc': unc, 'wavelength': wave, 'filter': filt, 'subarray': subarray}

    # Combine the order 1 and 2 spectra
    combined = []
    for n in range(nframes):
        spec1 = np.array([results['order1']['wavelength'], results['order1']['flux'][n], results['order1']['unc'][n]])
        spec2 = np.array([results['order2']['wavelength'], results['order2']['flux'][n], results['order2']['unc'][n]])
        spec3 = combine_spectra(spec1, spec2)
        combined.append(spec3)

    # Concatenate results and add to results
    wave_final = combined[0][0]
    flux_final = np.array([c[1] for c in combined])
    unc_final = np.array([c[2] for c in combined])
    results['final'] = {'wavelength': wave_final, 'counts': counts, 'flux': flux_final, 'unc': unc_final, 'filter': filt, 'subarray': subarray}

    return results

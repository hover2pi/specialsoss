# -*- coding: utf-8 -*-

"""A module of shared tools for SOSS data"""

import numpy as np


def combine_spectra(s1, s2):
    """
    Make a composite spectrum from two spectra

    Parameters
    ----------
    s1: sequence
        The [W, F, E] of the first spectrum
    s2: sequence
        The [W, F, E] of the second spectrum

    Returns
    -------
    sequence
        The [W, F, E] of the combined spectrum
    """
    # Determine if overlapping
    overlap = True
    try:
        if s1[0][-1] > s1[0][0] > s2[0][-1] or s2[0][-1] > s2[0][0] > s1[0][-1]:
            overlap = False
    except IndexError:
        overlap = False

    # Concatenate and order two segments if no overlap
    if not overlap:

        # Drop uncertainties on both spectra if one is missing
        if s1[2] is None or s2[2] is None:
            s1 = s1[:2]
            s2 = s2[:2]

        # Concatenate arrays and sort by wavelength
        new_spec = np.concatenate([s1, s2], axis=1).T
        new_spec = new_spec[np.argsort(new_spec[:, 0])].T

    # Otherwise there are three segments, (left, overlap, right)
    else:

        # Get the left segemnt
        left = s1[:, s1[0] <= s2[0][0]]
        if not np.any(left):
            left = s2[:, s2[0] <= s1[0][0]]

        # Get the right segment
        right = s1[:, s1[0] >= s2[0][-1]]
        if not np.any(right):
            right = s2[:, s2[0] >= s1[0][-1]]

        # Get the overlapping segements
        o1 = s1[:, np.where((s1[0] < right[0][0]) & (s1[0] > left[0][-1]))].squeeze()
        o2 = s2[:, np.where((s2[0] < right[0][0]) & (s2[0] > left[0][-1]))].squeeze()

        # Get the resolutions
        r1 = s1.shape[1]/(max(s1[0])-min(s1[0]))
        r2 = s2.shape[1]/(max(s2[0])-min(s2[0]))

        # Make higher resolution s1
        if r1 < r2:
            o1, o2 = o2, o1

        # Interpolate s2 to s1
        o2_flux = np.interp(o1[0], s2[0], s2[1])

        # Get the average
        o_flux = np.nanmean([o1[1], o2_flux], axis=0)

        # Calculate uncertainties if possible
        if len(s2) == len(o1) == 3:
            o2_unc = np.interp(o1[0], s2[0], s2[2])
            o_unc = np.sqrt(o1[2]**2 + o2_unc**2)
            overlap = np.array([o1[0], o_flux, o_unc])
        else:
            overlap = np.array([o1[0], o_flux])
            left = left[:2]
            right = right[:2]

        # Make sure it is 2D
        if overlap.shape == (3,):
            overlap.shape = 3, 1
        if overlap.shape == (2,):
            overlap.shape = 2, 1

        # Concatenate the segments
        new_spec = np.concatenate([left, overlap, right], axis=1)

    return new_spec

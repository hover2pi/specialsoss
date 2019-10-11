# -*- coding: utf-8 -*-

"""A module to correct GR700XD+CLEAR time series observations using GR700XD+F277W observations"""

from copy import copy

import numpy as np


def decontaminate(clear_data, f277w_data):
    """
    A function to decontaminate GR700XD+CLEAR spectra using GR700XD+F277W spectra

    Parameters
    ----------
    clear_data: np.ndarray
        The GR700XD+CLEAR datacube
    f277w_data: np.ndarray
        The GR700XD+F277W datacube
    """
    return clear_data

    # # Dict for decontaminated spctra
    # decontaminated = {}
    #
    # # Make sure the same extraction methods were used
    # for name in clear_data.keys():
    #     if name in f277w_data.keys():
    #
    #         # Get the data for each filter and order
    #         clear_ord1 = clear_data[name].get('order1')
    #         clear_ord2 = clear_data[name].get('order2')
    #         f277w_ord1 = f277w_data[name].get('order1')
    #
    #         # Corrected order 1 spectrum
    #         # corrected = copy()
    #
    #         # Correct as much contamination with the F277W spectrum as possible

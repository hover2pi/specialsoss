# -*- coding: utf-8 -*-

"""A module to correct GR700XD+CLEAR time series observations using GR700XD+F277W observations"""

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

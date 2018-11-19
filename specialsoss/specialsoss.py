# -*- coding: utf-8 -*-

"""A module to perform optimal spectral extraction of SOSS time series observations"""
import numpy as np


class SossObs:
    """A class object to extract and manipulate SOSS spectra"""
    def __init__(self, path, name='My SOSS Observations'):
        """Initialize the SOSS extraction object

        Parameters
        ----------
        path: str
            The path to the SOSS data
        name: str
            The name of the observation set
        """
        self.name = name
        self.path = path


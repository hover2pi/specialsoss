# -*- coding: utf-8 -*-

"""A module to perform optimal spectral extraction of SOSS time series observations"""

import os

from astropy.io import fits
# import jwst
import numpy as np

from . import crossdispersion as xdisp


class SossObs:
    """A class object to extract and manipulate SOSS spectra"""
    def __init__(self, filepath, name='My SOSS Observations', **kwargs):
        """Initialize the SOSS extraction object

        Parameters
        ----------
        path: str
            The path to the SOSS data
        name: str
            The name of the observation set
        """
        self.name = name
        self.filepath = filepath

        # Extract the data and header info
        self.run_pipeline(**kwargs)

    def run_pipeline(self, filepath=None, **kwargs):
        """Run the file through the JWST pipeline, storing the data and header

        Parameters
        ----------
        filepath: str
            The path to the file. If None, uses the filepath given at
            initialization
        """
        # Check if new filepath
        if filepath is None:
            filepath = self.filepath

        # Make sure the file exists
        if not os.path.exists(filepath) or not filepath.endswith('.fits'):
            raise FileError(filepath, ": Invalid file")

        # Get the data
        self.raw_data = fits.getdata(filepath)

        # Get the header and store cards as attributes
        self.header = fits.getheader(filepath)
        for card in self.header.cards:
            setattr(self, card[0], card[1])

        # Run the pipeline
        # self.datacube = jwst.run(filepath, **kwargs)

    def locate_trace(self, func=xdisp.bimodal, bounds=([15,0,15]*2, [110,5,np.inf]*2)):
        """Locate the traces in the data by convolving a function in each
        column

        Parameters
        ----------
        func: function
            The function to fit
        bounds: sequence
            The lower and upper bounds for the function input parameters
        """
        # Compose a median image from the stack
        median = np.median(self.datacube, axis=0)

        # Get a trace mask by fitting the given function
        # to each column in the median image
        self.mask = self.frame_mask(median, func, bounds)

        # Apply the median trace mask to each image to mask the whole cube
        mask_arr = np.array([self.mask]*len(self.datacube))
        self.maskedcube = ma.array(self.datacube, mask=mask_arr)

    @static_method
    def frame_mask(frame, func, bounds):
        """Generate a mask for the given frame"""
        pass
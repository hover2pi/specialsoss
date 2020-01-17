from copy import copy
import os

from astropy.io import fits
from bokeh.plotting import figure, show
from hotsoss import locate_trace as lt
from hotsoss import plotting as plt
from hotsoss import utils
import numpy as np


class SossFile:
    """
    A class object to handle JWST pipeline files for SOSS data
    """
    def __init__(self, filepath=None, **kwargs):
        """
        Initialize the SossFile object
        """
        # Acceptible levels
        self.levels = ['uncal', 'ramp', 'rate', 'rateints', 'calints', 'x1dints']

        # Set default attributes to None
        self._file = None
        self.ext = None
        self.hdulist = None
        self.header = None
        self.data = None
        self.nints = None
        self.ngrps = None
        self.nrows = None
        self.ncols = None
        self.filter = None
        self.subarray = None
        self.median = None
        self.wavecal = None

        # Set the filepath to populate the attributes
        self.file = filepath

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, filepath):
        """
        Setter for the file attribute

        Parameters
        ----------
        file: str
            The path to the file
        """
        if filepath is None:
            self._file = None

        elif isinstance(filepath, str):

            # Make sure the file exists
            if not os.path.exists(filepath):
                raise IOError("{} : Invalid file".format(filepath))

            # Determine processing level of input file
            if not any([filepath.endswith('_{}.fits'.format(level)) for level in self.levels]):
                raise ValueError('Not a recognized JWST file extension. Please use {}'.format(self.levels))

            # Save the filepath
            self._file = filepath

            # Save the filepath
            self.ext = filepath.split('_')[-1][:-5]

            # Get the header and data from the FITS file
            hdulist = fits.open(filepath)
            self.hdulist = copy(hdulist)
            self.header = hdulist['PRIMARY'].header
            self.data = hdulist['SCI'].data
            hdulist.close()

            # Observation parameters
            self.nints = self.header['NINTS']
            self.ngrps = self.header['NGROUPS']
            self.nrows = self.header['SUBSIZE2']
            self.ncols = self.header['SUBSIZE1']
            self.filter = self.header['FILTER']
            self.subarray = 'FULL' if self.nrows == 2048 else 'SUBSTRIP96' if self.nrows == 96 else 'SUBSTRIP256'

            # Compose a median image from the stack
            self.median = np.median(self.data, axis=(0, 1))

            # Load wavelength calibration
            self.load_wavecal()

        else:
            raise TypeError('File path must be a string. {} was given'.format(type(filepath)))

    def load_wavecal(self, file=None):
        """
        Load the wavelength calibration for orders 1, 2, and 3

        Parameters
        ----------
        file: str (optional)
            The path to a custom wavelength calibration file
        """
        # Load wavelength calibration file
        self.wavecal = utils.wave_solutions(subarray=self.subarray, file=file)

    def plot(self, scale='linear', coeffs=None, draw=True, **kwargs):
        """
        Plot the frames of data

        Parameters
        ----------
        scale: str
            The scale to plot, ['linear', 'log']
        draw: bool
            Draw the figure instead of returning it
        """
        # Reshape the data
        dim = self.data.shape
        if self.data.ndim == 4:
            data = self.data.reshape(dim[0]*dim[1], dim[2], dim[3])
        elif self.data.ndim == 2:
            data = self.data.reshape(1, dim[0], dim[1])
        else:
            data = self.data

        # Make the figure
        title = '{} Frames'.format(self.ext)
        coeffs = lt.trace_polynomial()
        fig = plt.plot_frames(data, idx=0, scale=scale, trace_coeffs=coeffs, wavecal=self.wavecal, title=title, **kwargs)

        if draw:
            show(fig)
        else:
            return fig

    def __repr__(self):
        """
        Return the path to the file
        """
        return self.file or 'None'

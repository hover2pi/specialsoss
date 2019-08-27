# -*- coding: utf-8 -*-

"""A module to perform optimal spectral extraction of SOSS time series observations"""

import copy
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from pkg_resources import resource_filename
import os

from astropy.io import fits
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap, log_cmap
from hotsoss import plotting as plt
import numpy as np

from . import crossdispersion as xdisp
from . import locate_trace as lt
from . import reconstruction as rc
from . import summation as sm
from . import binning as bn


def extract_flux(frame, coeffs=[lt.trace_polynomial(1), lt.trace_polynomial(2)], ):
    """
    Extract all spectral orders from a given frame

    Parameters
    ----------
    frame: array-like
        The 2D frame to extract a spectrum from
    """
    pass


class SossObs:
    """
    A class object to extract and manipulate SOSS spectra
    """
    def __init__(self, filepath, name='My SOSS Observations', process=True, **kwargs):
        """
        Initialize the SOSS extraction object

        Parameters
        ----------
        filepath: str
            The path to the SOSS data
        name: str
            The name of the observation set
        process: bool
            Pipeline process the input file
        """
        # Make sure the file exists
        if not os.path.exists(filepath) or not filepath.endswith('.fits'):
            raise FileNotFoundError("{}: Invalid file".format(filepath))

        # Store attributes
        self.name = name
        self.filepath = filepath
        self.time = None

        # Pipeline process
        if process:
            self.filepath = self._pipeline_process(filepath)

        # Load the file
        self.load_file(self.filepath)

        # Load the wavelength calibration and throughput curves
        self.load_filters()
        self.load_wavecal()
        self.load_wavebins()

        # Placeholder for the extracted spectra
        self.spectra = {}

    def caluclate_order_masks(self):
        """
        Calculate the order masks from the median image

        """
        # Find the trace in all columns
        self.order_masks = lt.order_masks(self.median, save=True)

        print("New order masks calculated from median image.")

    def extract(self, method="bin", **kwargs):
        """
        Extract the 1D spectra from the time series data
        using the specified method

        Parameters
        ----------
        method: str
            The method to use, ['reconstruct', 'wavebins']
        """
        # Validate the method
        valid_methods = ["reconstruct", "bin", "sum"]
        if method not in valid_methods:
            raise ValueError("{}: Not a valid extraction method. Please use {}".format(method, valid_methods))

        if method == "bin":
             self.spectra[method] = bn.extract(self.data)

        if method == "reconstruct":
            self.spectra[method] = rc.extract(self.data)

        if method == "sum":
            self.spectra[method] = sm.extract(self.data)


        # IDEAL
        # 1. Get the trace locations from the median frame and save ()
        # 2. Get the order masks from the traces and user input psf width (self.order_masks)
        # 3. 

    def _get_frame(self, idx=None):
        """
        Retrieve some frame data

        Parameters
        ----------
        idx: int
            The index of the frame to retrieve
        """
        if isinstance(idx, int):
            dim = self.data.shape
            frame = self.data.reshape(dim[0]*dim[1], dim[2], dim[3])[idx]
        else:
            frame = self.median

        return frame

    @property
    def info(self):
        """Return some info about the observation"""
        # Pull out relevant attributes
        track = ['ncols', 'nrows', 'nints', 'ngrps', 'subarray', 'filter']
        settings = {key: val for key, val in self.__dict__.items() if key in track}
        return settings

    def load_file(self, filepath, **kwargs):
        """
        Load the data and headers from an observation file

        Parameters
        ----------
        filepath: str
            The path to the SOSS data
        """
        # Make sure the file exists
        if not os.path.exists(filepath) or not filepath.endswith('.fits'):
            raise IOError(filepath, ": Invalid file")

        # Get the data
        self.raw_data = fits.getdata(filepath, **kwargs)
        self.header = fits.getheader(filepath)

        # Glean configuration info
        self.nints = self.header['NINTS']
        self.ngrps = self.header['NGROUPS']
        self.nrows = self.header['SUBSIZE2']
        self.ncols = self.header['SUBSIZE1']
        self.filter = self.header['FILTER']
        self.subarray = 'FULL' if self.nrows == 2048 else 'SUBSTRIP96' if self.nrows == 96 else 'SUBSTRIP256'

        # Ensure data is in 4 dimensions
        self.data = copy.copy(self.raw_data)
        if self.data.ndim == 3:
            self.data.shape = (self.nints, self.ngrps, self.nrows, self.ncols)
        if self.data.ndim != 4:
            raise ValueError("Data dimensions must be 3 or 4. {} is not valid.". format(self.raw_data.ndim))

        # Compose a median image from the stack
        self.median = np.median(self.data, axis=(0, 1))

        # Load the default order masks
        self.order_masks = lt.order_masks(self.median)

    def load_filters(self):
        """
        Load the wavelength bins for orders 1, 2, and 3
        """
        self.filters = []

        # Pull out the throughput for the appropriate order
        for ord in [1, 2, 3]:
            file = resource_filename('specialsoss', 'files/GR700XD_{}.txt'.format(ord))
            if os.path.isfile(file):
                self.filters.append(np.genfromtxt(file, unpack=True))

    def load_wavebins(self):
        """
        Load the wavelength bins for signal extraction
        """
        self.wavebins = lt.wavelength_bins()

    def load_wavecal(self, file=None):
        """
        Load the wavelength calibration for orders 1, 2, and 3

        Parameters
        ----------
        file: str (optional)
            The path to the wavelength calibration file
        """
        # Load default if None
        if file is None:
            file = resource_filename('specialsoss', 'files/soss_wavelengths_fullframe.fits')

        # Pull out the full frame data and trim for appropriate subarray
        start = 0 if self.subarray == 'FULL' else 1792
        end = 1888 if self.subarray == 'SUBSTRIP96' else 2048
        self.wavecal = fits.getdata(file).swapaxes(-2, -1)[:, start:end]

    def plot_frame(self, idx=None, scale='log', draw=True):
        """
        Plot a single frame of the data

        Parameters
        ----------
        idx: int
            The index of the frame to plot
        """
        # Get the data
        frame = self._get_frame(idx)

        # Make the figure
        c1 = lt.trace_polynomial(1)
        c2 = lt.trace_polynomial(2)
        title = 'Frame {}'.format(idx) if idx is not None else 'Median'
        fig = plt.plot_frame(frame, scale=scale, trace_coeffs=(c1, c2), title=title)

        if draw:
            show(fig)
        else:
            return fig

    def plot_slice(self, col, idx=None, draw=True, **kwargs):
        """
        Plot a column of a frame to see the PSF in the cross dispersion direction

        Parameters
        ----------
        col: int, sequence
            The column index(es) to plot
        idx: int
            The frame index to plot
        """
        # Get the data
        frame = self._get_frame(idx)

        # Plot the slice anf frame
        fig = plt.plot_slice(frame, col, idx=0, **kwargs)

        if draw:
            show(fig)
        else:
            return fig

    def plot_ramp(self, draw=True):
        """
        Plot the total flux on each frame to display the ramp
        """
        fig = plt.plot_ramp(self.data)

        if draw:
            show(fig)
        else:
            return fig

    @staticmethod
    def _pipeline_process(uncal_file, configdir=resource_filename('specialsoss', 'files/calwebb_tso1.cfg'), outdir=None):
        """
        Run the file through the JWST pipeline, storing the data and header
        """
        # Check for the pipeline
        try:

            if outdir is None:
                outdir = os.path.dirname(uncal_file)

            # Run the pipeline
            from jwst.pipeline import Detector1Pipeline
            tso1 = Detector1Pipeline.call(uncal_file, save_results=True, config_file=configdir, output_dir=outdir)
            processed = uncal_file.replace('.fits', '_ramp.fits')

            return processed

        except ImportError:
            print("Could not import JWST pipeline. {} has not been processed.".format(file))


class RealObs(SossObs):
    """
    A test instance with CV3 data loaded
    """
    def __init__(self, **kwargs):
        """
        Initialize the object
        """
        # Get the file
        file = resource_filename('specialsoss', 'files/SOSS256_CV3.fits')

        # Inherit from SossObs
        super().__init__(file, name='CV3 Observation', **kwargs)


class SimObs(SossObs):
    """
    A test instance with the data preloaded
    """
    def __init__(self, **kwargs):
        """
        Initialize the object
        """
        # Get the file
        file = resource_filename('specialsoss', 'files/SOSS256_sim.fits')

        # Inherit from SossObs
        super().__init__(file, name='Simulated Observation', **kwargs)

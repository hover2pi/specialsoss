# -*- coding: utf-8 -*-

"""A module to perform optimal spectral extraction of SOSS time series observations"""

import copy
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from pkg_resources import resource_filename
import os

from astropy.io import fits
from astropy import table as at
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap, log_cmap
from hotsoss import plotting as plt
from hotsoss import utils
from hotsoss import locate_trace as lt
import numpy as np

from . import reconstruction as rc
from . import summation as sm
from . import binning as bn


class SossObs:
    """
    A class object to extract and manipulate SOSS spectra
    """
    def __init__(self, filepath, name='My SOSS Observations', calibrate=True, **kwargs):
        """
        Initialize the SOSS extraction object

        Parameters
        ----------
        filepath: str
            The path to the SOSS data
        name: str
            The name of the observation set
        calibrate: bool
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
        if calibrate:
            self.filepath = self._pipeline_process(filepath)

        # Load the file
        self.load_file(self.filepath)

        # Load the wavelength calibration and throughput curves
        self.load_filters()
        self.load_wavecal()
        self.load_wavebins()

        # Placeholder for the extracted spectra
        cols = ('wavelength', 'counts', 'spectrum', 'method')
        dtypes = ('O', 'O', 'O', 'S12')
        self.extracted = at.Table(names=cols, dtype=dtypes)

    def caluclate_order_masks(self):
        """
        Calculate the order masks from the median image
        """
        # Find the trace in all columns
        self.order_masks = lt.order_masks(self.median, save=True)

        print("New order masks calculated from median image.")

    # def _counts_to_flux(self, wavelength, counts):
    #     """
    #     Convert the given count rate to a flux density
    #
    #     Parameters
    #     ----------
    #     wavelength: array-like
    #         The wavelength array
    #     counts: array-like
    #         The count rate as a function of wavelength
    #     """
    #     # Get the response for this order
    #
    #
    #
    #     # Get extracted spectrum (Column sum for now)
    #     wave = np.mean(self.wave[0], axis=0)
    #     flux_out = np.sum(tso.reshape(self.dims3)[frame].data, axis=0)
    #     response = 1./self.order1_response
    #
    #     # Convert response in [mJy/ADU/s] to [Flam/ADU/s] then invert so
    #     # that we can convert the flux at each wavelegth into [ADU/s]
    #     flux_out *= response/self.time[np.mod(self.ngrps, frame)]

    def extract(self, method="sum", **kwargs):
        """
        Extract the 1D spectra from the time series data
        using the specified method

        Parameters
        ----------
        method: str
            The extraction method to use, ["reconstruct", "bin", "sum"]
        """
        # Validate the method
        valid_methods = ["reconstruct", "bin", "sum"]
        if method not in valid_methods:
            raise ValueError("{}: Not a valid extraction method. Please use {}".format(method, valid_methods))

        # Make an entry dict
        entry = {'method': method}

        # Set the extraction function
        func = bn.extract if method == "bin" else sm.extract if method == "sum" else rc.extract

        # Run the extraction method, returning a dict
        # with keys ['counts', 'wavelength', 'flux']
        result = func(self.data, **kwargs)

        # Add the results to the table
        self.extracted.add_row(result)

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
            The path to a custom wavelength calibration file
        """
        # Load wavelength calibration file
        self.wavecal = utils.wave_solutions(subarray=self.subarray, file=file)

        # Pull out the full frame data and trim for appropriate subarray
        # end = 2048 if self.subarray == 'FULL' else 256
        # start = 160 if self.subarray == 'SUBSTRIP96' else 0
        # wave = fits.getdata(file).swapaxes(1, 2)[:, start:end, :]
        # wave[wave == 0] = np.nan
        # self.wavecal = wave

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

    def plot_frame(self, idx=None, scale='linear', draw=True):
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
        fig = plt.plot_frame(frame, scale=scale, trace_coeffs=(c1, c2), wavecal=self.wavecal, title=title)

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

        # Plot the slice and frame
        fig = plt.plot_slice(frame, col, idx=0, wavecal=self.wavecal, **kwargs)

        if draw:
            show(fig)
        else:
            return fig

    def plot_spectrum(self, idx=0, methods=['sum', 'bin', 'reconstruct'], draw=True):
        """
        Plot the extracted 1D spectrum

        Parameters
        ----------
        idx: int
            The frame index to plot
        """
        for method in methods:

            if method not in self.spectra:
                print("'{}' method not used for extraction. Skipping.".format(method))

            # Get the data
            spectrum = self.spectra[method]

            # Draw the figure
            fig = plt.plot_spectrum(spectrum)

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
    def __init__(self, calibrate=True, **kwargs):
        """
        Initialize the object
        """
        # To calibrate or not to calibrate
        ext = 'ramp' if calibrate else 'uncal'

        # Get the file
        file = resource_filename('specialsoss', 'files/SOSS256_sim_{}.fits'.format(ext))

        # Inherit from SossObs
        super().__init__(file, name='Simulated Observation', calibrate=False, **kwargs)

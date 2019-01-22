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
import numpy as np

from . import crossdispersion as xdisp
from . import locate_trace as lt


def extract_flux(frame, pixels):
    """Extract all spectral orders from a given frame

    Parameters
    ----------
    frame: array-like
        The 2D frame to extract a spectrum from
    """
    pass


class SossObs:
    """A class object to extract and manipulate SOSS spectra"""
    def __init__(self, filepath=None, name='My SOSS Observations', **kwargs):
        """Initialize the SOSS extraction object

        Parameters
        ----------
        filepath: str
            The path to the SOSS data
        name: str
            The name of the observation set
        """
        # Store attributes
        self.name = name
        self.filepath = filepath

        # Ingest data
        if filepath is not None:
            self.ingest_file(self.filepath, **kwargs)

        # Load order trace masks
        self.order_masks = np.zeros((3, 256, 2048))

        # Load the wavelength calibration and throughput curves
        self.load_filters()
        self.load_wavecal()
        self.load_wavebins()

        # Compose a median image from the stack
        self.median = np.median(self.datacube, axis=(0, 1))

        # Load the default order masks
        self.order_masks = lt.order_masks(self.median)

    def caluclate_order_masks(self):
        """Calculate the order masks from the median image
        """
        # Find the trace in all columns
        self.order_masks = lt.order_masks(self.median, save=True)

        print("New order masks calculated from median image.")

    def extract(self, n_jobs=4):
        """Extract the 1D spectrum from a frame

        Parameters
        ----------
        idx: int
            The index of the frame to extract

        Returns
        -------
        array
            The extracted 1D spectrum
        """
        # Multiprocess spectral extraction for frames
        pool = ThreadPool(n_jobs)
        func = partial(extract_spectrum, filters=self.filters, wavecal=self.wavecal)
        specs = pool.map(func, self.raw_data)
        pool.close()
        pool.join()

        self.tso = np.array(specs)

    def ingest_file(self, filepath, **kwargs):
        """Extract the data and header info from a file

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

        # Store header cards as attributes
        for card in self.header.cards:
            setattr(self, card[0], card[1])

        # The pipeline stores the data in shape (nints*ngrps, x, y)
        # Reshape into (nints, ngrps, x, y)
        self.datacube = copy.copy(self.raw_data)
        self.datacube.shape = (self.NINTS, self.NGROUPS, 2048, self.SUBSIZE2)
        self.datacube = self.datacube.swapaxes(-1, -2)

    def load_filters(self):
        """Load the wavelength bins for orders 1, 2, and 3
        """
        self.filters = []

        # Pull out the throughput for the appropriate order
        for ord in [1, 2, 3]:
            file = resource_filename('specialsoss', 'files/GR700XD_{}.txt'.format(ord))
            if os.path.isfile(file):
                self.filters.append(np.genfromtxt(file, unpack=True))

    def load_wavebins(self):
        """Load the wavelength bins for signal extraction"""
        self.wavebins = lt.wavelength_bins()

    def load_wavecal(self, file=None):
        """Load the wavelength calibration for orders 1, 2, and 3

        Parameters
        ----------
        file: str (optional)
            The path to the wavelength calibration file
        """
        # Load default if None
        if file is None:
            file = resource_filename('specialsoss', 'files/soss_wavelengths_fullframe.fits')

        # Pull out the data for the appropriate subarray
        self.wavecal = fits.getdata(file).swapaxes(-2, -1)[:, :self.SUBSIZE2]

    def plot_frame(self, idx=None, log_scale=True, draw=True):
        """Plot a single frame of the data

        Parameters
        ----------
        idx: int
            The index of the frame to plot
        """
        # Get the data
        if isinstance(idx, int):
            dim = self.datacube.shape
            frame = self.datacube.reshape(dim[0]*dim[1], dim[2], dim[3])[idx]
        else:
            frame = self.median

        # Make the figure
        fig = figure(x_range=(0, frame.shape[1]), y_range=(0, frame.shape[0]),
                     tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                     width=int(frame.shape[1]/2), height=int(frame.shape[0]/2),
                     title='Frame {}'.format(idx) if isinstance(idx, int) else 'Median Frame')

        # Plot the frame
        fig.image(image=[frame], x=0, y=0, dw=frame.shape[1],
                  dh=frame.shape[0], palette='Viridis256')

        # Plot the trace center
        fig.line(np.arange(2048), lt.trace_polynomial(1), color='red')
        fig.line(np.arange(2048), lt.trace_polynomial(2), color='red')

        if draw:
            show(fig)
        else:
            return fig

    # def run_pipeline(self, **kwargs):
    #     """Run the file through the JWST pipeline, storing the data and header
    #     """
    #     # Run the pipeline
    #     self.datacube = jwst.run(filepath, **kwargs)


class RealObs(SossObs):
    """A test instance with CV3 data loaded"""
    def __init__(self, **kwargs):
        """Initialize the object"""
        # Get the file
        file = resource_filename('specialsoss', 'files/SOSS256_CV3.fits')

        # Inherit from SossObs
        super().__init__(file, name='CV3 Observation', **kwargs)


class SimObs(SossObs):
    """A test instance with the data preloaded"""
    def __init__(self, **kwargs):
        """Initialize the object"""
        # Get the file
        file = resource_filename('specialsoss', 'files/SOSS256_sim.fits')

        # Inherit from SossObs
        super().__init__(file, name='Simulated Observation', **kwargs)

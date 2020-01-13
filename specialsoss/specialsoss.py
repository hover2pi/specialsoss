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

from . import decontaminate as dec
from . import reconstruction as rc
from . import summation as sm
from . import binning as bn


class SossExposure(object):
    """
    A class object to load a SOSS exposure file and extract and manipulate spectra
    """
    def __init__(self, filepath, name='My SOSS Observations', **kwargs):
        """
        Initialize the SOSS extraction object

        Parameters
        ----------
        filepath: str
            The path to the SOSS data
        name: str
            The name of the observation set
        """
        # Make sure the file exists
        if not os.path.exists(filepath) or not filepath.endswith('.fits'):
            raise FileNotFoundError("{}: Invalid file".format(filepath))

        # Store attributes
        self.name = name
        self.time = None

        # Determine processing level of input file
        if filepath.endswith('_uncal.fits'):
            self.uncal_file = filepath
            self.ramp_file = None
            self.calibrated = False

        elif filepath.endswith('_ramp.fits'):
            self.uncal_file = None
            self.ramp_file = filepath
            self.calibrated = True

        else:
            raise ValueError("{} processing level unclear. Please input an 'uncal' or 'ramp' file.".format(filepath))

        # Load the file
        self.load_file(filepath)

        # Load the wavelength calibration file
        self.load_filters()
        self.load_wavecal()

        # Dictionary for the extracted spectra
        self.extracted = {}

    def calibrate(self, **kwargs):
        """
        Pipeline process the current exposure
        """
        # No uncal file
        if self.uncal_file is None:
            print("No 'uncal' file to calibrate since 'ramp' file {} was initialized.".format(self.ramp_file))

        # Proceed with calibration
        else:
            self.ramp_file = self._pipeline_process(self.uncal_file, **kwargs)

            # Load the calibrated ramp file
            self.load_file(self.ramp_file)

    def caluclate_order_masks(self):
        """
        Calculate the order masks from the median image
        """
        # Find the trace in all columns
        self.order_masks = lt.order_masks(self.median, save=True)

        print("New order masks calculated from median image.")

    def decontaminate(self, f277w_exposure):
        """
        Decontaminate the GR700XD+CLEAR orders with GR700XD+F277W order 1

        Parameters
        ----------
        f277w_exposure: SossExposure
            The F277W exposure object
        """
        # Check that the current filter is CLEAR
        if not self.filter == 'CLEAR':
            raise ValueError("filter = {}: Only a CLEAR exposure can be decontaminated".format(self.filter))

        # Check the dtype
        if not isinstance(f277w_exposure, type(self)):
            raise TypeError("{}: f277w_exposure must be of type {}".format(type(f277w_exposure), type(self)))

        # Check that the new filter is F277W
        new_filt = f277w_exposure.filter
        if new_filt != 'F277W':
            raise ValueError("filter = {}: Only an F277W exposure can be used for decontamination".format(new_filt))

        # Check that spectral extraction has been run on the CLEAR exposure
        if not bool(self.extracted):
            raise ValueError("Please run 'extract' method on CLEAR exposure before decontamination")

        # Check that spectral extraction has been run on the F277W exposure
        if not bool(f277w_exposure.extracted):
            raise ValueError("Please run 'extract' method on F277W exposure before decontamination")

        # Run the decontamination
        self.extracted = dec.decontaminate(self.extracted, f277w_exposure.extracted)

    def extract(self, method="sum", name=None, **kwargs):
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

        # Set the extraction function
        func = bn.extract if method == "bin" else sm.extract if method == "sum" else rc.extract

        # Run the extraction method, returning a dict with keys ['counts', 'wavelength', 'flux']
        result = func(self.data, filt=self.filter, subarray=self.subarray, **kwargs)
        result['method'] = method

        # Add the results to the table
        if name is None:
            name = method
        self.extracted[name] = result

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
            if self.data.ndim == 4:
                frame = self.data.reshape(dim[0]*dim[1], dim[2], dim[3])[idx]
            else:
                frame = self.data[idx]
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
            raise IOError("{} : Invalid file".format(filepath))

        # Glean configuration info
        self.header = fits.getheader(filepath)
        self.nints = self.header['NINTS']
        self.ngrps = self.header['NGROUPS']
        self.nrows = self.header['SUBSIZE2']
        self.ncols = self.header['SUBSIZE1']
        self.filter = self.header['FILTER']
        self.subarray = 'FULL' if self.nrows == 2048 else 'SUBSTRIP96' if self.nrows == 96 else 'SUBSTRIP256'

        # Ensure data is in 4 dimensions
        self.data = fits.getdata(filepath, **kwargs)
        if self.data.ndim == 3:
            self.data.shape = (self.nints, self.ngrps, self.nrows, self.ncols)
        if self.data.ndim != 4:
            raise ValueError("Data dimensions must be 3 or 4. {} is not valid.". format(self.data.ndim))

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
            file = resource_filename('hotsoss', 'files/GR700XD_{}.txt'.format(ord))
            if os.path.isfile(file):
                self.filters.append(np.genfromtxt(file, unpack=True))

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
            processed = uncal_file.replace('_uncal.fits', '_ramp.fits')

            return processed

        except ImportError:
            print("Could not import JWST pipeline. {} has not been processed.".format(file))

    def plot_frame(self, idx=None, scale='linear', draw=True, **kwargs):
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
        title = '{}: Frame {}'.format(self.name, idx if idx is not None else 'Median')
        coeffs = lt.trace_polynomial()
        fig = plt.plot_frame(frame, scale=scale, trace_coeffs=coeffs, wavecal=self.wavecal, title=title, **kwargs)

        if draw:
            show(fig)
        else:
            return fig

    def plot_frames(self, idx=0, scale='linear', draw=True, **kwargs):
        """
        Plot a single frame of the data

        Parameters
        ----------
        idx: int
            The index of the frame to plot
        """
        # Reshape the data
        dim = self.data.shape
        data = self.data.reshape(dim[0]*dim[1], dim[2], dim[3])

        # Make the figure
        title = '{}: Frames'.format(self.name)
        coeffs = lt.trace_polynomial()
        fig = plt.plot_frames(data, idx=idx, scale=scale, trace_coeffs=coeffs, wavecal=self.wavecal, title=title, **kwargs)

        if draw:
            show(fig)
        else:
            return fig

    def plot_extracted_spectra(self, name=None, draw=True):
        """
        Plot the extracted 1D spectra

        Parameters
        ----------
        name: str (optional)
            The name of the extracted data
        """
        # Select the extractions
        fig = None
        if name is None:
            name = self.extracted.keys()[0]

        # Get the data dictionary and color
        result = self.extracted[name]

        # Draw the figure
        counts = result['order1']['counts']
        flux = result['order1']['flux']
        wave = result['order1']['wavelength']
        fig = plt.plot_time_series_spectra(wave=wave, flux=flux)

        if draw:
            show(fig)
        else:
            return fig

    def plot_spectra(self, idx=0, dtype='flux', names=None, order=None, draw=True):
        """
        Plot the extracted 1D spectrum

        Parameters
        ----------
        idx: int
            The frame index to plot
        dtype: str
            The data to plot, ['flux', 'counts']
        names: seqence (optional)
            The names of the extracted spectra to plot
        order: int (optional)
            The order to plot
        """
        # Get colors palette
        colors = utils.COLORS

        # Select the orders
        if isinstance(order, int):
            orders = [order]
        else:
            orders = [1, 2]

        if dtype == 'counts':
            ylabel = 'Counts [ADU/s]'
        else:
            ylabel = 'Flux Density [erg/s/cm2/A]'

        # Select the extractions
        fig = None
        if names is None:
            names = self.extracted.keys()

        for name in names:

            if name not in self.extracted:
                print("'{}' method not used for extraction. Skipping.".format(name))

            else:

                # Get the data dictionary and color
                result = self.extracted[name]
                color = next(colors)

                # Draw the figure
                data = result[dtype]
                wave = result['wavelength']
                flux = data[idx]
                fig = plt.plot_spectrum(wave, flux, fig=fig, legend=name, ylabel=ylabel, color=color, alpha=0.8)

                # # Draw the figure with orders separated
                # for order in orders:
                #     key = 'order{}'.format(order)
                #     if key in result:
                #         legend = ' - '.join([key, name])
                #         data = result[key][dtype]
                #         wave = result[key]['wavelength']
                #         flux = data[idx]
                #         fig = plt.plot_spectrum(wave, flux, fig=fig, legend=legend, ylabel=ylabel, color=color, alpha=1./order)

        if fig is not None:
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


class SimExposure(SossExposure):
    """
    A test instance with the data preloaded
    """
    def __init__(self, subarray='SUBSTRIP256', filt='CLEAR', level='uncal', **kwargs):
        """
        Initialize the SOSS extraction object

        Parameters
        ----------
        subarray: str
            The desired subarray, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
        filt: str
            The desired filter, ['CLEAR', 'F277W']
        level: str
            The desired level of pipeline processing, ['uncal', 'ramp']
        """
        # Get the file
        file = resource_filename('specialsoss', 'files/{}_{}_{}.fits'.format(subarray, filt, level))

        # Inherit from SossObs
        super().__init__(file, name='Simulated Observation', **kwargs)

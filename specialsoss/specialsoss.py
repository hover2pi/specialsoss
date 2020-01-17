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
from . import sossfile as sf


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
            The path to the SOSS data, which must end with '_<ext>.fits',
            where 'ext' is ['uncal', 'ramp', 'rate', 'rateints', 'calints', 'x1dints']
        name: str
            The name of the exposure set
        """
        # Make sure the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError("{}: Invalid file".format(filepath))

        # Store attributes
        self.name = name
        self.time = None

        # Store empty SossFile objects
        self.uncal = sf.SossFile()  # 4D Uncalibrated raw input
        self.ramp = sf.SossFile() # 4D Corrected ramp data
        self.rate = sf.SossFile() # 2D Corrected countrate image
        self.rateints = sf.SossFile() # 3D Corrected countrate per integration
        self.calints = sf.SossFile() # 3D Calibrated data
        self.x1dints = sf.SossFile() # 1D Extracted spectra

        # Reset file levels
        self.levels = ['uncal', 'ramp', 'rate', 'rateints', 'calints', 'x1dints']

        # Load the file
        self.load_file(filepath)

        # Load the order throughput, wavelength calibration, and order mask files
        self.load_filters()
        # self.order_masks = lt.order_masks(self.median)

        # Dictionary for the extracted spectra
        self.extracted = {}

        # Print uncal warning
        if self.uncal is not None:
            print("Looks like you have initialized an 'uncal' file! To pipeline process it, run 'SossExposure.uncal.calibrate()' method.")

    def calculate_order_masks(self):
        """
        Calculate the order masks from the median image
        """
        # Find the trace in all columns
        self.order_masks = lt.order_masks(self.median, save=True)

        print("New order masks calculated from median image.")

    def calibrate(self, ext='uncal', configdir=None, outdir=None, **kwargs):
        """
        Pipeline process a file in the exposure object

        Parameters
        ----------
        ext: str
            The extension to calibrate
        configdir: str
            The directory containing the configuration files
        outdir: str
            The directory to put the calibrated files into
        """
        if ext not in self.levels:
            raise ValueError("'{}' not valid extension. Please use {}".format(ext, self.levels))

        # Plot the appropriate file
        fileobj = getattr(self, ext)
        if fileobj.file is not None:
            new_files = fileobj.calibrate(configdir=configdir, outdir=outdir, **kwargs)

        # Load the calibrated files
        for level, file in new_files.items():
            self.load_file(file)

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

    @property
    def info(self):
        """Return some info about the observation"""
        # Pull out relevant attributes
        track = ['ncols', 'nrows', 'nints', 'ngrps', 'subarray', 'filter']
        settings = {key: val for key, val in self.__dict__.items() if key in track}

        # Get file info
        for level in self.levels:
            fileobj = getattr(self, level)
            file = None if fileobj is None else fileobj.file
            settings.update({level: file})

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
        if not os.path.exists(filepath):
            raise IOError("{} : Invalid file".format(filepath))

        # Determine processing level of input file
        if not any([filepath.endswith('_{}.fits'.format(level)) for level in self.levels]):
            raise ValueError('Not a recognized JWST file extension. Please use {}'.format(self.levels))

        # Save the filepath
        ext = filepath.split('_')[-1][:-5]
        fileobj = sf.SossFile(filepath)
        setattr(self, ext, fileobj)

        # Load the attributes
        self.nints = fileobj.nints
        self.ngrps = fileobj.ngrps
        self.nrows = fileobj.nrows
        self.ncols = fileobj.ncols
        self.filter = fileobj.filter
        self.subarray = fileobj.subarray
        self.wavecal = fileobj.wavecal

        print("'{}' file loaded from {}".format(ext, filepath))

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

    def plot(self, ext='uncal', idx=None, scale='linear', draw=True, **kwargs):
        """
        Plot a frame or all frames of any image data

        Parameters
        ----------
        ext: str
            The extension to plot
        idx: int
            The index of the frame to plot
        """
        if ext not in self.levels:
            raise ValueError("'{}' not valid extension. Please use {}".format(ext, self.levels))

        # Plot the appropriate file
        fileobj = getattr(self, '{}_file'.format(ext))
        if fileobj.file is not None:
            fileobj.plot(idx=idx, scale=scale, draw=draw)

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

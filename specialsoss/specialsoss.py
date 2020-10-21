# -*- coding: utf-8 -*-

"""A module to perform optimal spectral extraction of SOSS time series observations"""

from functools import wraps
import os
from pkg_resources import resource_filename

from bokeh.plotting import show
from bokeh.layouts import column
from hotsoss import plotting as plt
from hotsoss import utils
from hotsoss import locate_trace as lt
import numpy as np

from . import decontaminate as dec
from . import summation as sm
from . import binning as bn
from . import halftrace as ht
from . import jetspec as jc
from . import sossfile as sf


def results_required(func):
    """A wrapper to check that the extraction has been run before a method can be executed"""
    @wraps(func)
    def _results_required(*args, **kwargs):
        """Check that the 'results' dictionary is not empty"""
        if not bool(args[0].results):
            print("No extraction found! Please run the 'extract' method first.")

        else:
            return func(*args, **kwargs)

    return _results_required


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

        # Dictionary for the extracted spectra
        self.results = {}

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
        self.order_masks = lt.order_masks(self.median)

        # Print uncal warning
        if self.uncal.file is not None:
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

    @results_required
    def compare_results(self, idx=0, dtype='counts', names=None, order=None, draw=True):
        """
        Compare results of multiple extraction routines for a given frame

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
        fig1, fig2 = None, None
        if names is None:
            names = self.results.keys()

        for name in names:

            if name not in self.results:
                print("'{}' method not used for extraction. Skipping.".format(name))

            else:

                # Get the data dictionary and color
                result = self.results[name]
                color = next(colors)

                # Get the order 1 data
                order1_result = result['order1']
                order1_counts = order1_result['counts']
                order1_unc = order1_result['unc']
                order1_wave = order1_result['wavelength']

                # Get the order 2 data
                order2_result = result['order2']
                order2_counts = order2_result['counts']
                order2_unc = order2_result['unc']
                order2_wave = order2_result['wavelength']

                # Draw the figures
                x = 'Wavelength [um]'
                y = dtype.upper()
                fig1 = plt.plot_spectrum(order1_wave, order1_counts[idx], fig=fig1, legend=name, ylabel=ylabel, color=color, alpha=0.8)
                fig2 = plt.plot_spectrum(order2_wave, order2_counts[idx], fig=fig2, legend=name, ylabel=ylabel, color=color, alpha=0.8)

        if draw:
            show(column([fig1, fig2]))
        else:
            return fig1, fig2

    @results_required
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

        # Check that spectral extraction has been run on the F277W exposure
        if not bool(f277w_exposure.results):
            raise ValueError("Please run 'extract' method on F277W exposure before decontamination")

        # Run the decontamination
        self.results = dec.decontaminate(self.results, f277w_exposure.results)

    def extract(self, method="sum", ext='rateints', name=None, **kwargs):
        """
        Extract the 1D spectra from the time series data
        using the specified method

        Parameters
        ----------
        method: str
            The extraction method to use, ["bin", "sum"]
        ext: str
            The extension to extract
        name: str
            A name for the extraction results
        """
        # Validate the method
        valid_methods = ["bin", "sum", "jetspec", "halftrace"]
        if method not in valid_methods:
            raise ValueError("{}: Not a valid extraction method. Please use {}".format(method, valid_methods))

        # Set the extraction function
        mod = bn if method == "bin" else jc if method == "jetspec" else ht if method == "halftrace" else sm

        # Get the requested data
        fileobj = getattr(self, ext)
        if fileobj.data is None:
            raise ValueError("No '{}' data to extract.".format(ext))

        # Run the extraction method, returning a dict with keys ['counts', 'wavelength', 'flux']
        result = mod.extract(fileobj.data, filt=self.filter, subarray=self.subarray, **kwargs)
        result['method'] = method

        # Add the results to the table
        if name is None:
            name = method
        self.results[name] = result

    def _get_extension(self, ext, err=True):
        """
        Get the data for the given extension

        Parameters
        ----------
        ext: str
            The name of the extension
        err: bool
            Throw an error instead of printing
        """
        # Check the level is valid
        if ext not in self.levels:
            raise ValueError("'{}' not valid extension. Please use {}".format(ext, self.levels))

        # Get the data from the appropriate file
        fileobj = getattr(self, ext)
        if fileobj.file is None:
            loaded = [level for level in self.levels if getattr(self, level).file is not None]
            msg = "No data for '{0}' extension. Load `_{0}.fits' file or try {1}".format(ext, loaded)

            if err:
                raise ValueError(msg)
            else:
                print(msg)

        else:
            return fileobj

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
        self.nframes = fileobj.nframes
        self.nrows = fileobj.nrows
        self.ncols = fileobj.ncols
        self.filter = fileobj.filter
        self.frame_time = fileobj.frame_time
        self.subarray = fileobj.subarray
        self.wavecal = fileobj.wavecal
        self.median = fileobj.median
        self.time = fileobj.time

        # Load the awesimsoss input spectrum as a "result" for direct comparison
        if fileobj.star is not None:
            wave, flux = fileobj.star
            counts = np.ones((fileobj.nframes, len(wave))) * np.nan
            flux = np.array([flux] * fileobj.nframes)
            unc = np.ones_like(counts)
            self.results['input'] = {'order1': {'wavelength': wave, 'flux': flux, 'counts': counts, 'unc': unc, 'filter': 'None', 'subarray': 'None', 'method': 'None'},
                                     'order2': {'wavelength': wave, 'flux': flux, 'counts': counts, 'unc': unc, 'filter': 'None', 'subarray': 'None', 'method': 'None'}}

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

    def plot_frames(self, ext='uncal', scale='linear', draw=True, **kwargs):
        """
        Plot a frame or all frames of any image data

        Parameters
        ----------
        ext: str
            The extension to plot
        idx: int
            The index of the frame to plot
        draw: bool
            Draw the figure instead of returning it
        """
        # Get the file object
        fileobj = self._get_extension(ext)

        # Make the plot
        fig = fileobj.plot(scale=scale)

        if draw:
            show(fig)
        else:
            return fig

    # # From awesimsoss!!
    # @run_required
    # def plot_lightcurve(self, column, time_units='s', resolution_mult=20, draw=True):
    #     """
    #     Plot a lightcurve for each column index given
    #
    #     Parameters
    #     ----------
    #     column: int, float, sequence
    #         The integer column index(es) or float wavelength(s) in microns
    #         to plot as a light curve
    #     time_units: string
    #         The units for the time axis, ['s', 'min', 'h', 'd' (default)]
    #     resolution_mult: int
    #         The number of theoretical points to plot for each data point
    #     draw: bool
    #         Render the figure instead of returning it
    #     """
    #     # Check time_units
    #     if time_unit not in ['s', 'min', 'h', 'd']:
    #         raise ValueError("time_unit must be 's', 'min', 'h' or 'd']")
    #
    #     # Get the scaled flux in each column for the last group in
    #     # each integration
    #     flux_cols = np.nansum(self.tso_ideal.reshape(self.dims3)[self.ngrps - 1::self.ngrps], axis=1)
    #     flux_cols = flux_cols / np.nanmax(flux_cols, axis=0)[None, :]
    #
    #     # Make it into an array
    #     if isinstance(column, (int, float)):
    #         column = [column]
    #
    #     # Make the figure
    #     lc = figure()
    #
    #     for kcol, col in enumerate(column):
    #
    #         color = next(utils.COLORS)
    #
    #         # If it is an index
    #         if isinstance(col, (int, np.integer)):
    #             lightcurve = flux_cols[:, col]
    #             label = 'Column {}'.format(col)
    #
    #         # Or assumed to be a wavelength in microns
    #         elif isinstance(col, float):
    #             waves = np.mean(self.wave[0], axis=0)
    #             lightcurve = [np.interp(col, waves, flux_col) for flux_col in flux_cols]
    #             label = '{} um'.format(col)
    #
    #         else:
    #             print('Please enter an index, astropy quantity, or array thereof.')
    #             return
    #
    #         # Plot the theoretical light curve
    #         if str(type(self.tmodel)) == "<class 'batman.transitmodel.TransitModel'>":
    #
    #             # Make theoretical time axis
    #             time = self.duration * np.arange(self.nframes * resolution_mult)
    #
    #             # Generate transit model with time in [days] for batman
    #             tmodel = batman.TransitModel(self.tmodel, time.to(q.d))
    #             tmodel.rp = self.rp[col]
    #             theory = tmodel.light_curve(tmodel)
    #             theory *= max(lightcurve) / max(theory)
    #
    #             # Convert time axis to desired units
    #             time = time.to(time_units).value
    #
    #             # Add to figure
    #             lc.line(time, theory, legend=label+' model', color=color, alpha=0.8)
    #
    #         # Convert data time axis to desired units
    #         data_time = TimeDelta(self.time - self.time.min()).to(time_units).value
    #
    #         # Plot the lightcurve
    #         lc.circle(data_time, lightcurve, legend=label, color=color)
    #
    #     lc.xaxis.axis_label = 'Time [{}]'.format(time_units)
    #     lc.yaxis.axis_label = 'Transit Depth'
    #
    #     if draw:
    #         show(lc)
    #     else:
    #         return lc
    #
    # @run_required
    # def plot_spectrum(self, frame=0, order=None, noise=False, scale='log', draw=True):
    #     """
    #     Plot a column sum of counts converted to flux density as a function of the mean column wavelength
    #
    #     Parameters
    #     ----------
    #     frame: int
    #         The frame number to plot
    #     order: sequence
    #         The order to isolate
    #     noise: bool
    #         Plot with the noise model
    #     scale: str
    #         Plot scale, ['linear', 'log']
    #     draw: bool
    #         Render the figure instead of returning it
    #     """
    #     # Get the data cube
    #     tso = self._select_data(order, noise)
    #
    #     # Get counts per wavelength (Column sum for now)
    #     wave = np.mean(self.wave[0], axis=0)
    #     counts = np.sum(tso[frame].data, axis=0)
    #     response = self.order1_response
    #
    #     # Convert response in [mJy/ADU/s] to [Flam/ADU/s] then invert so
    #     # that we can convert the flux at each wavelegth into [ADU/s]
    #     flux_out = (counts / response / (self.time[np.mod(self.ngrps, frame)]*q.s)).value
    #
    #     # Trim wacky extracted edges
    #     flux_out[0] = flux_out[-1] = np.nan
    #
    #     # Plot it along with input spectrum
    #     flux_in = np.interp(wave, self.star[0], self.star[1])
    #
    #     # Make the spectrum plot
    #     spec = figure(x_axis_type=scale, y_axis_type=scale, width=1024, height=500)
    #     spec.step(wave, flux_out, mode='center', legend='Extracted', color='red')
    #     spec.step(wave, flux_in, mode='center', legend='Injected', alpha=0.5)
    #     spec.yaxis.axis_label = 'Flux Density [{}]'.format(self.star[1].unit)
    #
    #     # Get the residuals
    #     res = figure(x_axis_type=scale, x_range=spec.x_range, width=1024, height=150)
    #     res.step(wave, flux_out - flux_in, mode='center')
    #     res.xaxis.axis_label = 'Wavelength [{}]'.format(self.star[0].unit)
    #     res.yaxis.axis_label = 'Residuals'
    #
    #     if draw:
    #         show(column(spec, res))
    #     else:
    #         return column(spec, res)

    @results_required
    def plot_results(self, name=None, dtype='counts', time_fmt='mjd', draw=True):
        """
        Plot results of all integrations for the given extraction routine

        Parameters
        ----------
        name: str (optional)
            The name of the extracted data to plot
        dtype: str
            The data type to plot, ['flux', 'counts']
        time_fmt: str
            The astropy time format to use
        draw: bool
            Draw the figure instead of returning it
        """
        # Check dtype
        dtypes = ['flux', 'counts']
        if dtype not in dtypes:
            raise ValueError("{}: Please select dtype from {}".format(dtype, dtypes))

        # Check name
        names = list(self.results.keys())
        if name is not None and name not in names:
            raise ValueError("{}: Name not in results. Try {}".format(name, names))

        # Select the extractions
        fig = None
        if name is None:
            name = names[0]

        # Get the data for the named extraction method
        result = self.results[name]

        # Get the order 1 data
        order1_result = result['order1']
        order1_counts = order1_result['counts']
        order1_unc = order1_result['unc']
        order1_wave = order1_result['wavelength']

        # Get the order 2 data
        order2_result = result['order2']
        order2_counts = order2_result['counts']
        order2_unc = order2_result['unc']
        order2_wave = order2_result['wavelength']

        # Get time array
        time = self.time.to_value(time_fmt)

        # Draw the figure
        x = 'Wavelength [um]'
        y = 'Time [{}]'.format(time_fmt.upper())
        fig1 = plt.plot_time_series_spectra(order1_counts, wavelength=order1_wave, time=time, ylabel=y, xlabel=x)
        fig2 = plt.plot_time_series_spectra(order2_counts, wavelength=order2_wave, time=time, ylabel=y, xlabel=x)

        if draw:
            show(column([fig1, fig2]))
        else:
            return fig1, fig2


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
            The desired level of pipeline processing, ['uncal', 'rateints']
        """
        # Get the file
        file = resource_filename('specialsoss', 'files/{}_{}_{}.fits'.format(subarray, filt, level))

        # Inherit from SossObs
        super().__init__(file, name='Simulated Observation', **kwargs)

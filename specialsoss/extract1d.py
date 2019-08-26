# -*- coding: utf-8 -*-

"""A module of 1D spectral extraction routines"""

import copy

from awesimsoss import TSO, make_trace as mt
from astropy.io import fits
import astropy.units as q
from bokeh.plotting import figure, show
import lmfit
import numpy as np
from pkg_resources import resource_filename

from . import locate_trace as lt


def calc_spectrum(**scales):
    """Calculate a spectrum by column sum"""
    # Make the spectrum
    file = resource_filename('specialsoss', 'files/soss_wavelengths_fullframe.fits')
    wave = np.mean(fits.getdata(file).swapaxes(-2, -1)[0, :, 1792:2048], axis=1)[::-1]*q.um
    flux = np.array([v for k, v in scales.items()])*q.erg/q.s/q.cm**2/q.AA
    frame = make_frame([wave, flux])
    spectrum = np.sum(frame, axis=0)

    return spectrum

def make_frame(spectrum, plot=False):
    """Make a frame with N psfs with a pixel width and a pixel offset for each psf
    where each psf is scaled randomly"""
    # Make a TSO object
    sim = TSO(nints=1, ngrps=1, star=spectrum, verbose=False)
    sim.simulate(noise=False)

    # Grab the frame
    frame = sim.tso[0, 0]

    if plot:
        # Plot the frame
        sim.plot(idx=0)

    return frame

def lmfitter(frame, model=calc_spectrum, uncertainty=None, method='powell', verbose=True, **kwargs):
    """Use lmfit to find the scale of each psf

    Parameters
    ----------
    frame: sequence
        The frame data
    model: function
        The model to fit

    Returns
    -------
    lmfit.Model.fit.fit_report
        The results of the fit
    """
    # Get a rough guess for the flux by summing the columns
    # TODO: This is actually a count rate and needs to be converted to a flux density before summing
    flux = np.sum(frame, axis=0)

    # Initialize lmfit Params object
    initialParams = lmfit.Parameters()

    # Make each column a parameter with the sum as the starting guess
    param_list = [('w{}'.format(n), wav, True) for n, wav in enumerate(flux)]

    # Set independent variables
    indep_vars = {}

    # Get values from input parameters.Parameters instances
    initialParams.add_many(*param_list)

    # Create the lightcurve model
    specmodel = lmfit.Model(model)
    specmodel.independent_vars = indep_vars.keys()

    # Set the uncertainty
    if uncertainty is None:
        uncertainty = np.ones_like(flux)

    # Fit model to the simulated data
    result = specmodel.fit(flux, weights=1/uncertainty, params=initialParams, method=method, **indep_vars, **kwargs)

    # Plot the original spectrum
    fig = figure()
    fig.line(x, flux, legend='Input Spectrum', color='blue')

    # And the fit spectrum
    fit = calc_spectrum(**result.best_values)
    fig.line(x, fit, legend='Best Fit Spectrum', color='green')
    show(fig)

    # Plot residuals
    fig = figure()
    fig.line(x, flux-fit, color='red')
    show(fig)

    # Plot the frame diff
    fit_frame = make_frame(**result.best_values)
    fig = figure(tooltips=[('x', '$x'), ('y', '$y'), ("value", "@image")])
    data = dict(image=[frame])
    fig.image(image=[frame-fit_frame], x=0, y=0, dw=frame.shape[0], dh=frame.shape[1], palette="Inferno256")
    show(fig)

    # Return results
    return result

# sosspsf = mt.get_SOSS_psf(1, filt='CLEAR', psfs=None, cutoff=0)

# def psf(scale, wavelength=1, pix=76, xy=50):
#     """Make a psf of a given size and scale"""
#     return copy.copy(sosspsf)*scale
#
#
# def make_frame(n_psfs, offset, pix, xy, plot=False, **scales):
#     """Make a frame with N psfs with a pixel width and a pixel offset for each psf
#     where each psf is scaled randomly"""
#     # Set the values
#     frame = np.zeros((76, 76+n_psfs))
#
#     # Make the frame
#     for n in range(n_psfs):
#         frame[:, n:n+pix] += psf(scales['w{}'.format(n)], pix=pix)
#
#     # Trim off excess
#     frame = frame[:, 38:-38]
#
#     if plot:
#         # Plot the frame
#         fig = figure(tooltips=[('x', '$x'), ('y', '$y'), ("value", "@image")])
#         data = dict(image=[frame])
#         fig.image(source=data, x=0, y=0, image='image', dw=frame.shape[0], dh=frame.shape[1], palette="Inferno256")
#         show(fig)
#
#     return frame
#
#
# def calc_spectrum(n_psfs, offset, pix, xy, plot=False, **scales):
#     """Calculate a spectrum by column sum"""
#     frame = make_frame(n_psfs, offset, pix, xy, plot=plot, **scales)
#     spectrum = np.sum(frame, axis=0)
#
#     return spectrum
#
#
# def lmfitter(frame, model=calc_spectrum, uncertainty=None, method='powell', verbose=True, **kwargs):
#     """Use lmfit to find the scale of each psf
#
#     Parameters
#     ----------
#     frame: sequence
#         The frame data
#     model: function
#         The model to fit
#
#     Returns
#     -------
#     lmfit.Model.fit.fit_report
#         The results of the fit
#     """
#     # Make a spectrum
#     spectrum = np.sum(frame, axis=0)
#     x = np.arange(len(spectrum))
#
#     # Initialize lmfit Params object
#     initialParams = lmfit.Parameters()
#
#     # Concatenate the lists of parameters
#     param_list = [('w{}'.format(n), 1., True) for n in range(kwargs['n_psfs'])]
#     param_list.append(('offset', kwargs['offset'], False))
#     param_list.append(('pix', kwargs['pix'], False))
#     param_list.append(('xy', kwargs['xy'], False))
#     param_list.append(('n_psfs', kwargs['n_psfs'], False))
#
#     # Set independent variables
#     indep_vars = {}
#
#     # Get values from input parameters.Parameters instances
#     initialParams.add_many(*param_list)
#
#     # Create the lightcurve model
#     specmodel = lmfit.Model(model)
#     specmodel.independent_vars = indep_vars.keys()
#
#     # Set the uncertainty
#     if uncertainty is None:
#         uncertainty = np.ones_like(spectrum)
#
#     # Fit model to the simulated data
#     result = specmodel.fit(spectrum, weights=1/uncertainty, params=initialParams, method=method, **indep_vars, **kwargs)
#
#     # Plot the original spectrum
#     fig = figure()
#     fig.line(x, spectrum, legend='Input Spectrum', color='blue')
#
#     # And the fit spectrum
#     fit = calc_spectrum(**result.best_values)
#     fig.line(x, fit, legend='Best Fit Spectrum', color='green')
#     show(fig)
#
#     # Plot residuals
#     fig = figure()
#     fig.line(x, spectrum-fit, color='red')
#     show(fig)
#
#     # Plot the frame diff
#     fit_frame = make_frame(**result.best_values)
#     fig = figure(tooltips=[('x', '$x'), ('y', '$y'), ("value", "@image")])
#     data = dict(image=[frame])
#     fig.image(image=[frame-fit_frame], x=0, y=0, dw=frame.shape[0], dh=frame.shape[1], palette="Inferno256")
#     show(fig)
#
#     # Return results
#     return result

# def place_psf_shift(psf, coords, frame, plot=False):
#     """Place the given 2D *psf* on a frame of shape *frame_shape* with a center at *coords*
#
#     Parameters
#     ----------
#     psf: array-like
#         The 2D psf to place
#     coords: tuple
#         The (x, y) coordinates of frame to place the psf center
#     frame: array-like
#         The frame to place the psf on
#     plot: bool
#         Plot the figure
#     """
#     # Make a psf as large as the frame
#     data = np.zeros(frame.shape)
#     data[:psf.shape[0], :psf.shape[1]] = psf[:, :]
#
#     # Subtract half the distance as the bottom left is at 0,0 instead of the center.
#     x_0 = (psf.shape[0] - 1.) / 2.
#     y_0 = (psf.shape[1] - 1.) / 2.
#     data = shift(data, (coords[1] - y_0, coords[0] - x_0))
#
#     frame += data
#
#     if plot:
#         plot_frame(frame, coords=coords)
#
# def place_psf_spline(psf, coords, frame, plot=False):
#     """Place the given 2D *psf* on a frame of shape *frame_shape* with a center at *coords*
#
#     Parameters
#     ----------
#     psf: array-like
#         The 2D psf to place
#     coords: tuple
#         The (x, y) coordinates of frame to place the psf center
#     frame: array-like
#         The frame to place the psf on
#     plot: bool
#         Plot the figure
#     """
#     # A gaussian to add to the frame
#     x = np.arange(psf.shape[0], dtype=np.float)
#     y = np.arange(psf.shape[1], dtype=np.float)
#     spline = RectBivariateSpline(x, y, psf.T, kx=3, ky=3, s=0)
#
#     # Define coordinates of a feature in the data array.
#     # This can be the center of the Gaussian:
#     x_0 = (psf.shape[0] - 1.0) / 2.0
#     y_0 = (psf.shape[1] - 1.0) / 2.0
#
#     # create output grid, shifted as necessary:
#     xg, yg = np.indices(frame.shape, dtype=np.float64)
#     xg += x_0 - coords[0]
#     yg += y_0 - coords[1]
#
#     # resample and fill extrapolated points with 0:
#     resampled_psf = spline.ev(xg, yg).T
#     # extrapol = (((xg < -0.5) | (xg >= psf.shape[1] - 0.5)) | ((yg < -0.5) | (yg >= psf.shape[0] - 0.5)))
#     # resampled_psf[extrapol] = 0
#
#     resampled_psf += frame.T
#
#     if plot:
#         plot_frame(resampled_psf, coords=coords)
#
#     return resampled_psf
#
#
# def plot_frame(frame, coords=None):
#     """Plot a frame with or without crosshairs"""
#     fig = figure(x_range=(0, frame.shape[0]), y_range=(0, frame.shape[1]), height=frame.shape[1], width=frame.shape[0], tooltips=[('x', '$x'), ('y', '$y'), ("value", "@image")])
#     data = dict(image=[frame])
#     fig.image(source=data, x=0, y=0, image='image', dw=frame.shape[0], dh=frame.shape[1], palette="Inferno256")
#
#     if coords is not None:
#         fig.line(np.arange(frame.shape[0]), coords[1]*np.ones(frame.shape[0]), color='white', alpha=0.2)
#         fig.line(coords[0]*np.ones(frame.shape[1]), np.arange(frame.shape[1]), color='white', alpha=0.2)
#
#     show(fig)
#
# def test_place_psf(shape, coords):
#     """A test for the place_psf function, """
#     # Make the frame
#     frm = np.random.normal(size=shape, scale=0.1)
#
#     # Make the psf
#     # xx, yy = np.meshgrid(np.linspace(-1, 1, 76), np.linspace(-1, 1, 76))
#     # d = np.sqrt(xx*xx + yy*yy)
#     # sigma, mu = 0.25, 0.0
#     # psf = 2*np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
#     psf = 100*mt.get_SOSS_psf(1, filt='CLEAR', psfs=None, cutoff=0)
#
#     # Subtract
#     interp = place_psf(psf, coords, frm.copy(), plot=True)

# def test_place_soss_trace():
#     psf = 100*mt.get_SOSS_psf(1, filt='CLEAR', psfs=None, cutoff=0)
#     frame = np.zeros((300,200))
#     for x in range(10):
#         frame = place_psf(psf, coords, frame)
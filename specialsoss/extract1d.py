# -*- coding: utf-8 -*-

"""A module of 1D spectral extraction routines"""

import copy

import lmfit
import numpy as np

from . import make_trace as smt


def lmfitter_soss(frame, model, uncertainty=None, method='powell', verbose=True, **kwargs):
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
    # Make a spectrum
    spectrum = np.sum(frame, axis=0)
    x = np.arange(len(spectrum))
    
    # Initialize lmfit Params object
    initialParams = lmfit.Parameters()

    # Concatenate the lists of parameters
    param_list = [('w{}'.format(n), 1., True) for n in range(n_psfs)]
    param_list.append(('offset', offset, False))
    param_list.append(('pix', pix, False))
    param_list.append(('xy', xy, False))
    param_list.append(('n_psfs', n_psfs, False))

    # Set independent variables
    indep_vars = {}

    # Get values from input parameters.Parameters instances
    initialParams.add_many(*param_list)

    # Create the lightcurve model
    specmodel = lmfit.Model(model)
    specmodel.independent_vars = indep_vars.keys()

    # Set the uncertainty
    if uncertainty is None:
#         uncertainty = np.ones_like(frame)
        uncertainty = np.ones_like(spectrum)

    # Fit model to the simulated data
    # result = specmodel.fit(frame, weights=1/uncertainty, params=initialParams, method=method, **indep_vars, **kwargs)
    result = specmodel.fit(spectrum, weights=1/uncertainty, params=initialParams, method=method, **indep_vars, **kwargs)
    
    # Plot the original spectrum
    fig = figure()
    fig.line(x, spectrum, legend='Input Spectrum', color='blue')
    
    # And the fit spectrum
    fit = calc_spectrum_soss(**result.best_values)
    fig.line(x, fit, legend='Best Fit Spectrum', color='green')
    show(fig)
    
    # Plot residuals
    fig = figure()
    fig.line(x, spectrum-fit, color='red')
    show(fig)
    
    # Plot the frame diff
    fit_frame = make_frame_soss(**result.best_values)
    fig = figure(tooltips=[('x', '$x'), ('y', '$y'), ("value", "@image")])
    data = dict(image=[frame])
    fig.image(image=[frame-fit_frame], x=0, y=0, dw=frame.shape[0], dh=frame.shape[1], palette="Inferno256")
    show(fig)
    
    # Return results
    return result
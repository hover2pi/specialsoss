"""Module to locate the signal traces in SOSS 2D frames"""
import os
import multiprocessing
import time
import warnings

from bokeh.plotting import figure, show
import numpy as np
from functools import partial
from numpy import ma
from scipy.optimize import curve_fit

from . import crossdispersion as xdsp

warnings.simplefilter('ignore')


def function_traces(data2D, start1=(4,75), start2=(520,145), xdisp=20, offset=0.5, **kwargs):
    """
    Return the given data with non-trace pixels masked
    
    Parameters
    ==========
    data2D: array-like
        The 2D data to search for a trace
    start1: tuple
        The (row,column) to start looking for order 1
    start2: tuple
        The (row,column) to start looking for order 2
    xdisp: int
        The radius of the psf in the cross-dispersion direction
    plot: bool
        Plot the masked 2D data
        
    Returns
    =======
    np.ndarray
        The given 2D data with non-signal pixels masked
        
    """
    # Make a masked array of the 2D data
    masked_data1 = ma.array(data2D.copy(), mask=False)
    masked_data2 = ma.array(data2D.copy(), mask=False)
    
    # Set initial constraints
    llim1 = start1[1]-xdisp-5
    ulim1 = start1[1]+xdisp+5
    llim2 = start2[1]-xdisp-5
    ulim2 = start2[1]+xdisp+5
        
    # Replace each column mask with the one calculated from isolate_signal,
    # using the mask of the previous column as curve_fit constraints
    for n,col in enumerate(masked_data1):
            
        # Update constraints
        if n > 0:
            
            # For order 1
            try:
                if llim1 < 5:
                    llim1 = 5
                    ulim1 = 5+xdisp*2
                elif np.where(~masked_data1.mask[n-1])[0][0] < llim1:
                    llim1 -= offset
                    ulim1 -= offset
                else:
                    llim1 += offset
                    ulim1 += offset
                    
            except:
                pass
                
            # For order 2
            try:
                
                if np.where(~masked_data2.mask[n-1])[0][0] < llim2:
                    llim2 -= offset
                    ulim2 -= offset
                else:
                    llim2 += offset
                    ulim2 += offset
                    
            except:
                pass
                
        # Only extract order 1
        if n > start1[0] and n < start2[0]:
            
            try:
                bounds = ([llim1, 1, 5, 1, 5, 3], [ulim1, 5, 2500, 10, 2500, 30])
                col_mask1 = isolate_signal(col, func=batman, xdisp=xdisp, bounds=bounds)
            except:
                col_mask1 = ma.array(col, mask=masked_data1.mask[n-1])
            
            col_mask2 = ma.array(col, mask=True)
            
        # Extract both orders
        elif n > start1[0] and n > start2[0]:
            
            try:
                bounds = ([llim1, 2, 5, 2, 5, 3, llim2, 2, 3, 2, 3],
                          [ulim1, 5, 2500, 10, 2500, 30, ulim2, 4, 10, 8, 10])
                col_mask1, col_mask2 = isolate_orders(col, bounds=bounds, xdisp=xdisp)
            except:
                col_mask1 = ma.array(col, mask=masked_data1.mask[n-1])
                col_mask2 = ma.array(col, mask=masked_data2.mask[n-1])
                
        # Don't extract either order
        else:
            col_mask1 = ma.array(col, mask=True)
            col_mask2 = ma.array(col, mask=True)
            
        # Set the mask in the image
        masked_data1.mask[n] = col_mask1.mask
        masked_data2.mask[n] = col_mask2.mask

    # Smooth the trace
    smooth_masked_data1 = smooth_trace(masked_data1, 4, xdisp, start1[0])
    try:
        smooth_masked_data2 = smooth_trace(masked_data2, 3, xdisp, start2[0])
    except:
        smooth_masked_data2 = ''

    # Plot it
    plt.figure(figsize=(13,2))
    plt.imshow(masked_data1.data, origin='lower', norm=LogNorm())
    plt.imshow(smooth_masked_data1, origin='lower', norm=LogNorm(), cmap=plt.cm.Blues_r, alpha=0.7)
    try:
        plt.imshow(smooth_masked_data2, origin='lower', norm=LogNorm(), cmap=plt.cm.Reds_r, alpha=0.7)
    except:
        pass
    plt.xlim(0, 2048)
    plt.ylim(0, 256)
    plt.show()
        
    return smooth_masked_data1, smooth_masked_data2

def smooth_trace(masked_data, order, width, start, plot=True):
    """
    Smooth a trace by fitting a polynomial to the unmasked pixels
    
    Parameters
    ----------
    masked_data: numpy.ma.array
        The masked data to smooth
    order: int
        The polynomial order to fit to the unmasked data
    width: int
        The desired radius of the trace from the best fit line
    start: int
        The index of the first pixel to fit
    plot: bool
        Plot the original image with the smoothed fits
    
    Returns
    -------
    numpy.ma.array
        The smoothed masked array 
    """
    return masked_data
    # Make a scatter plot where the pixels in each column are offset by a small amount
    # x, y = [], []
    # for n, col in enumerate(masked_data):
    #     vals = np.where(~col.mask)
    #     if vals:
    #         v = list(vals[0])
    #         y += v
    #         x += list(np.random.normal(n, 0.01, size=len(v)))
    #
    # # plt.imshow(masked_data.data, origin='lower', norm=LogNorm())
    # plt.scatter(x, y)
    # # plt.fill_between(X, bottom, top, color='b', alpha=0.3)
    # plt.xlim(0,2048)
    # plt.ylim(0,256)
    # plt.show()
    #
    # # Now fit a polynomial to it!
    # height, length = masked_data.shape[-1], masked_data.shape[-2]
    # Y = np.polyfit(x[start:], y[start:], order)
    # X = np.arange(start, length, 1)
    #
    # # Get pixel values of trace and top and bottom bounds
    # trace = np.floor(np.polyval(Y, X)).round().astype(int)
    # bottom = trace-width
    # top = trace+width
    #
    # # Plot it
    # if plot:
    #     plt.imshow(masked_data.data, origin='lower', norm=LogNorm())
    #     plt.plot(X, trace, color='b')
    #     plt.fill_between(X, bottom, top, color='b', alpha=0.3)
    #     plt.xlim(0,length)
    #     plt.ylim(0,height)
    #
    # # Create smoothed mask from top and bottom polynomials
    # smoothed_data = ma.array(masked_data.data, mask=True)
    # for n,(b,t) in enumerate(zip(bottom,top)):
    #     smoothed_data.mask[n+start][b:t] = False
    #
    # return smoothed_data

def trace_polynomial(order):
    """The polynomial that describes the order trace

    Parameters
    ----------
    order: int
        The order polynomial

    Returns
    -------
    sequence
        The y values of the given order across the 2048 pixels
    """
    coeffs = [[1.71164994e-11, -4.72119272e-08, 5.10276801e-05, -5.91535309e-02, 8.30680347e+01],
              [2.35792131e-13, 2.42999478e-08, 1.03641247e-05, -3.63088657e-02, 9.96766537e+01]]

    return np.polyval(coeffs[order-1], np.arange(2048))


def isolate_orders(frame):
    """
    Return the given data with non-trace pixels masked
    
    Parameters
    ==========
    frame: array-like
        The 2D data to search for a trace
        
    Returns
    =======
    np.ndarray
        The given 2D data with non-signal pixels masked
        
    """
    # Make a masked array of the 2D data
    masked_data1 = ma.array(frame.copy(), mask=False)
    masked_data2 = ma.array(frame.copy(), mask=False)

    # Replace each column mask with the one calculated from isolate_signal,
    # using the mask of the previous column as curve_fit constraints
    for n,col in enumerate(masked_data1):
            
        # Update constraints
        if n > 0:
            
            # For order 1
            try:
                if llim1 < 5:
                    llim1 = 5
                    ulim1 = 5+xdisp*2
                elif np.where(~masked_data1.mask[n-1])[0][0] < llim1:
                    llim1 -= offset
                    ulim1 -= offset
                else:
                    llim1 += offset
                    ulim1 += offset
                    
            except:
                pass
                
            # For order 2
            try:
                
                if np.where(~masked_data2.mask[n-1])[0][0] < llim2:
                    llim2 -= offset
                    ulim2 -= offset
                else:
                    llim2 += offset
                    ulim2 += offset
                    
            except:
                pass
                
        # Only extract order 1
        if n > start1[0] and n < start2[0]:
            
            try:
                bounds = ([llim1, 1, 5, 1, 5, 3], [ulim1, 5, 2500, 10, 2500, 30])
                col_mask1 = isolate_signal(col, func=batman, xdisp=xdisp, bounds=bounds)
            except:
                col_mask1 = ma.array(col, mask=masked_data1.mask[n-1])
            
            col_mask2 = ma.array(col, mask=True)
            
        # Extract both orders
        elif n > start1[0] and n > start2[0]:
            
            try:
                bounds = ([llim1, 2, 5, 2, 5, 3, llim2, 2, 3, 2, 3],
                          [ulim1, 5, 2500, 10, 2500, 30, ulim2, 4, 10, 8, 10])
                col_mask1, col_mask2 = isolate_orders(col, bounds=bounds, xdisp=xdisp)
            except:
                col_mask1 = ma.array(col, mask=masked_data1.mask[n-1])
                col_mask2 = ma.array(col, mask=masked_data2.mask[n-1])
                
        # Don't extract either order
        else:
            col_mask1 = ma.array(col, mask=True)
            col_mask2 = ma.array(col, mask=True)
            
        # Set the mask in the image
        masked_data1.mask[n] = col_mask1.mask
        masked_data2.mask[n] = col_mask2.mask

    # Smooth the trace
    smooth_masked_data1 = smooth_trace(masked_data1, 4, xdisp, start1[0])
    try:
        smooth_masked_data2 = smooth_trace(masked_data2, 3, xdisp, start2[0])
    except:
        smooth_masked_data2 = ''

    # Plot it
    plt.figure(figsize=(13,2))
    plt.imshow(masked_data1.data, origin='lower', norm=LogNorm())
    plt.imshow(smooth_masked_data1, origin='lower', norm=LogNorm(), cmap=plt.cm.Blues_r, alpha=0.7)
    try:
        plt.imshow(smooth_masked_data2, origin='lower', norm=LogNorm(), cmap=plt.cm.Reds_r, alpha=0.7)
    except:
        pass
    plt.xlim(0, 2048)
    plt.ylim(0, 256)
    plt.show()
        
    return smooth_masked_data1, smooth_masked_data2


def isolate_signal(idx, frame, bounds=None, sigma=3, err=None, radius=None, filt='CLEAR', plot=True):
    """
    Fit a mixed gaussian function to the signal in an array of data. 
    Identify all pixels within n-sigma as signal.
    
    Parameters
    ----------
    idx: int
        The index of the column in the 2048 pixel wide subarray
    frame: array-like
         The 2D frame to pull the column from
    err: array-like (optional)
        The errors in the 1D data
    bounds: tuple
        A sequence of length-n (lower,upper) bounds on the n-parameters of func
    sigma: float
        The number of standard deviations to use when defining the signal
    plot: bool
        Plot the signal with the fit function
    
    Returns
    -------
    np.ndarray
        The values of signal pixels and the upper and lower bounds on the fit function
    """
    # Get the column of data
    col = frame[:, idx]
    col[120:] *= 400

    # Use the trace centers as the position of the center peak
    x1 = trace_polynomial(1)[idx]
    x2 = trace_polynomial(2)[idx]

    # Set the column at which order 2 ends
    order2end = 1900 if col.size == 256 else 1050

    # Two cases where there is no order 2
    if (idx >= order2end and filt == 'CLEAR') or filt == 'F277W':

        # Use the batman function to find only the first order
        func = xdsp.batman

        # Same as above, just one signal
        bounds = ([x1-3, 2, 100, 2, 300, 5], [x1+3, 4, 1e6, 8, 1e6, 10])

    # Otherwise there are two orders
    else:

        # Use the batmen function to find both orders
        func = xdsp.batmen

        # Set (lower, upper) bounds for each parameter
        # --------------------------------------------
        # x-position of the center peak, second psf
        # stanfard deviation of the center peak, second psf
        # amplitude of the center peak, second psf
        # stanfard deviation of the outer peaks, second psf
        # amplitude of the outer peaks, second psf
        # separation of the outer peaks from the center, second psf
        # x-position of the center peak, first psf
        # stanfard deviation of the center peak, first psf
        # amplitude of the center peak, first psf
        # stanfard deviation of the outer peaks, first psf
        # amplitude of the outer peaks, first psf
        # separation of the outer peaks from the center, first psf
        bounds = ([x2-3, 2, 3, 2, 3, 5, x1-3, 2, 100, 2, 300, 5],
                  [x2+3, 4, 1e3, 8, 1e3, 10, x1+3, 4, 1e6, 8, 1e6, 10])

    # Fit function to signal
    col = ma.array(col, mask=False)
    x = ma.array(np.arange(len(col)))
    params, cov = curve_fit(func, x, col, bounds=bounds, sigma=err)

    # -----------------------------------------------------------------------
    # Order 1
    # -----------------------------------------------------------------------
    # Reduce to mixed gaussians with arguments (mu, sigma, A)
    p1 = params[:3]
    p2 = [params[0]-params[5], params[3], params[4]]
    p3 = [params[0]+params[5], params[3], params[4]]

    # Get the mu, sigma, and amplitude of each gaussian in each order
    params1 = np.array(sorted(np.array([p1,p2,p3]), key=lambda x: x[0]))

    # If radius is given use a set number of pixels as the radius
    if isinstance(radius, int):
        llim1 = params1[1][0]-radius
        ulim1 = params1[1][0]+radius

    # Otherwise, use the sigma value
    else:
        llim1 = params1[0][0]-params1[0][1]*sigma
        ulim1 = params1[-1][0]+params1[-1][1]*sigma

    # Make a mask for oredr 1
    ord1 = ma.array(col)
    ord1.mask[(x > llim1) & (x < ulim1)] = True
    ord1.mask = np.invert(ord1.mask)

    # -----------------------------------------------------------------------
    # Order 2
    # -----------------------------------------------------------------------
    if func == xdsp.batmen:

        # Reduce to mixed gaussians with arguments (mu, sigma, A)
        p4 = params[6:9]
        p5 = [params[6]-params[11], params[9], params[10]]
        p6 = [params[6]+params[11], params[9], params[10]]

        # Get the mu, sigma, and amplitude of each gaussian in each order
        params2 = np.array(sorted(np.array([p4,p5,p6]), key=lambda x: x[0]))

        # If radius is given use a set number of pixels as the radius
        if isinstance(radius, int):
            llim2 = params2[1][0]-radius
            ulim2 = params2[1][0]+radius

        # Otherwise, use the sigma
        else:
            llim2 = params2[0][0]-params2[0][1]*sigma
            ulim2 = params2[-1][0]+params2[-1][1]*sigma

        # Make a mask for order 2
        ord2 = ma.array(col)
        ord2.mask[(x > llim2) & (x < ulim2)] = True
        ord2.mask = np.invert(ord2.mask)

    # Mask the whole column for order 2
    else:
        p4 = p5 = p6 = None
        ord2 = ma.array(col, mask=True)

    # -----------------------------------------------------------------------

    if plot:

        # Make the figure
        fig = figure(x_range=(0, col.size), width=1000, height=600,
                     tooltips=[("x", "$x"), ("y", "$y"), ("value", "@line")],
                     title='Column {}'.format(idx))

        # The data
        fig.line(x, col, color='black', legend='Data')

        # The fit function and components
        fig.line(x, func(x, *params), color='red', legend='Fit')
        for g in [p1, p2, p3, p4, p5, p6]:
            if g is not None:
                fig.line(x, xdsp.gaussian(x, *g), alpha=0.5, legend='Components')

        show(fig)

    return ord1, ord2

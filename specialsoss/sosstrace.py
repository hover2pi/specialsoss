"""Module to locate the signal traces in SOSS 2D frames"""

from copy import copy
from functools import partial
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import time
import warnings

from astropy.modeling import models, fitting
from bokeh.layouts import column, row
from bokeh.models import CustomJS, Slider, LogColorMapper, Range1d, CrosshairTool, Div
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from scipy.interpolate import CubicSpline, UnivariateSpline
import numpy as np

warnings.simplefilter('ignore')


def trace_fit(data, orders=[1, 2, 3], F277W=False, tolerance=10):
    """
    Find the traces in all columns for a set of frames

    Parameters
    ----------
    data: sequence
        The 4D data to fit traces to of shape (nints, ngrps, rows, cols)
    orders: sequence
        The order traces to fit
    F277W: bool
        Just fit order 1 F277W
    tolerance: float
        The tolerance for detecting a shifted trace position

    Returns
    -------
    np.ndarray
        The y-position of each trace in each group of shape (nints, ngrps, orders, cols)

    """
    # Data dimensions
    nints, ngrps, nrows, ncols = data.shape

    # Empty array for results
    results = np.zeros((nints, ngrps, 3, 2048))

    # Just check trace in the last group of each integration
    last_groups = data[:, -1, :, :]

    # And first group to make sure the position hasn't changed in the first integration
    last_groups = np.concatenate([[data[0, 0, :, :]], last_groups])

    # Run multiprocessing
    pool = ThreadPool(cpu_count())
    func = partial(trace_polynomials, orders=orders, F277W=F277W, verbose=False)
    last_results = pool.map(func, last_groups)
    pool.close()
    pool.join()
    del pool

    print(last_results[0].shape)

    # Put the last group results in the final results array
    results[0, 0, :, :] = last_results[0]
    results[:, -1, :, :] = last_results[1:]

    # Function to check for changes in the trace
    def trace_check(trace1, trace2, tolerance=tolerance):
        return np.sum(np.abs(trace1 - trace2)) > tolerance

    # Check for changes across integrations
    for nint in np.arange(nints):

        # For first integration, check between first and last group
        # For all other integrations, check between last group of current and next integration
        first = results[0, 0] if nint == 0 else results[nint, -1]
        last = results[0, -1] if nint == 0 else results[nint + 1, -1]

        # Check to see if the trace position changed, given tolerance
        int_ok = trace_check(first, last, tolerance=tolerance)

        # If the trace position hasn't changed, interpolate to get the trace positions in between
        if int_ok:
            results[0, 1:-2, :, :] = np.interp(np.arange(ncols), first, last)

    return results

def rateint_trace_fit(data, orders=[1, 2, 3], F277W=False, tolerance=10, plot=True):
    """
    Find the traces in all columns for a set of frames

    Parameters
    ----------
    data: sequence
        The 3D data to fit traces to of shape (nints, rows, cols)
    orders: sequence
        The order traces to fit
    F277W: bool
        Just fit order 1 F277W
    tolerance: float
        The tolerance for detecting a shifted trace position

    Returns
    -------
    np.ndarray
        The y-position of each trace in each group of shape (nints, orders, cols)

    """
    # Tell me about it
    print('Extracting trace positions of orders {} from {} data of shape {}'.format('1' if F277W else orders, 'F277W' if F277W else 'CLEAR', data.shape))
    start = time.time()

    # Run multiprocessing
    pool = ThreadPool(cpu_count())
    func = partial(trace_polynomials, orders=orders, F277W=F277W, verbose=False)
    int_results = pool.map(func, data)
    pool.close()
    pool.join()
    del pool

    print('Trace finder finished: {} {}'.format(round(time.time() - start, 3), 's'))

    return int_results


def get_counts(params, col):
    """
    Get the number of counts from the given column of data and best fit parameters

    Parameters
    ----------
    params: dict
        The best fit parameters for the column of data
    col: sequence
        The column of raw data that was fit

    Returns
    -------
    sequence
        The counts in the first and second order traces
    """
    # Generate the cross-dispersed PSF given the best fit parameters
    order1 = np.sum(functional_psf(params, order=1), axis=0)
    order2 = np.sum(functional_psf(params, order=2), axis=0)
    order3 = np.sum(functional_psf(params, order=3), axis=0)

    # TODO: Perform some correction to reproduce the actual measured counts in *col*
    # TODO: Order3 can be found by just setting an extraction aperture
    # TODO: Order1 and Order2 can be found by multiplying the total measured counts in
    # TODO: a column by the ratio of the order1 and order2 functional_psf values

    # Integrate the functional form of the PSF to get the counts
    counts1 = np.trapz(order1)
    counts2 = np.trapz(order2)
    counts3 = np.trapz(order3)

    # Show residuals of functional form vs. actual
    func_counts = counts1 + counts2 + counts3
    true_counts = np.sum(col)
    residual = true_counts - func_counts

    # Print results
    print('Functional counts:', func_counts)
    print('True counts:', true_counts)
    print('Residual:', residual)

    return counts1, counts2, counts3, func_counts, true_counts, residual


def trace_polynomials(frame, skip=1, ref_pix=4, F277W=False, order2=True, order3=True, order1start=None, verbose=True, plot=False, **kwargs):
    """
    Derive the trace polynomials for orders 1, 2, snd 3 from scratch. Takes about 5 minutes.

    Parameters
    ----------
    frame:
        The frame of data to analyze
    start: int
        The y-position of the order 1 trace in column 1
    """
    # Transpose frame
    frame = frame.T

    # Time it
    if verbose:
        start = time.time()
        print("Starting trace finder...")

    # Stop fitting order 1 at this column index
    stop_1 = 440 if F277W else 2048

    # Stop fitting order 2 at this column index
    stop_2 = 0 if (F277W or not order2) else 1050 if frame.shape[1] == 96 else 1800

    # Stop fitting order 3 at this column
    stop_3 = 0 if (frame.shape[1] == 96 or F277W or not order3) else 800

    # Find the fit parameters in an uncontaminated column then scan left to right
    start_col = stop_1 if F277W else 1200

    # List of columns to check
    check_cols = np.arange(0, 2048, skip)

    # Iterate over columns to the right (unless F277W)
    if F277W:
        start_params = fit_psfs(frame[start_col], ord12_x0=order1start or 68, order2=False, order3=False)
        results_right = []

    else:
        start_params = fit_psfs(frame[start_col], ord12_x0=order1start or 43, ord22_x0=(order1start or 43) + 72, order3=False)
        start_params['col_num'] = start_col
        results_right = [start_params]
        params = start_params
        for idx, col in enumerate(frame[start_col + 1:-ref_pix]):

            # Check if order 2 and 3 are valid
            col_num = np.arange(2048)[start_col + 1 + idx]
            order2 = col_num <= stop_2

            if col_num in check_cols:

                # Perform fit
                params = fit_psfs(col, order2=order2, order3=False, plot=False, **params)
                params['col_num'] = col_num

                # Append results
                results_right.append(params)

    # Iterate over columns to the left
    results_left = []
    params = start_params
    for idx, col in enumerate(frame[ref_pix:min(stop_1, start_col)][::-1]):

        # Check if order 2 and 3 are valid
        col_num = np.arange(stop_1)[start_col - idx - 1]
        order2 = col_num <= stop_2

        if col_num in check_cols:

            # Constrain order 2 more between col 0 and 600
            if col_num < 600:
                center = np.interp(col_num, [0, 600], [105, params['ord22_x0']])
                ord22_x0_min = center - 1
                ord22_x0_max = center + 1
            else:
                ord22_x0_min = ord22_x0_max = None

            # Perform fit
            params = fit_psfs(col, order2=order2, order3=False, plot=False, ord22_x0_min=ord22_x0_min, ord22_x0_max=ord22_x0_max, **params)
            params['col_num'] = col_num

            # Append results
            results_left.append(params)

    # Reverse since we scanned the columns backwards
    results_left = results_left[::-1]

    # Fit order 3 separately
    start_params = fit_psfs(frame[ref_pix + 1], order1=False, order2=False, ord32_x0=(order1start or 51) + 96, plot=False)
    params = start_params
    for idx, col in enumerate(frame[ref_pix + 1:stop_3]):

        # Check if order 2 and 3 are valid
        col_num = ref_pix + idx

        if col_num in check_cols:

            # Perform fit
            params = fit_psfs(col, order1=False, order2=False, plot=False, **params)
            params['col_num'] = col_num

            # Update left results
            just_order3 = {k: v for k, v in params.items() if 'ord3' in k or k in ['sep3']}
            results_left[col_num].update(just_order3)

    # Concatenate results
    results = results_left + results_right

    # Guide the landing on order 2 if the trace is too faint
    for n, res in enumerate(results[:600]):
        default = np.interp(n, [0, 600], [105, 97])
        if default - 2 < res['ord22_x0'] > default + 1:
            results[n]['ord22_x0'] = default

    # Guide the landing on the blue end of order 2
    for n, res in enumerate(results[1780:1796]):
        results[res['col_num']]['ord22_x0'] = np.interp(res['col_num'], [1780, 1796], [results[1780]['ord22_x0'], 256])

    # Add ref pixels
    ref_cols = [np.nan] * ref_pix

    # Plot
    image = copy(frame)
    image[image < 1.] = 1.
    mapper = LogColorMapper(palette='Viridis256', low=np.nanmin(image), high=np.nanmax(image))
    fig = figure(width=900, height=200, x_range=(0, 2048), y_range=(0, 256))
    img = image.T
    img_nans = np.where(np.isnan(img))
    img[img_nans] = 0
    fig.image(image=[img.astype(np.float64)], x=0, y=0, dh=256, dw=2048, color_mapper=mapper)

    # Fit polynomial to order 1 centers
    o1_x0 = np.array(ref_cols + [i['ord12_x0'] for i in results] + ref_cols)
    o1_nums = np.argwhere(~np.isnan(o1_x0)).flatten()
    o1_x0 = o1_x0[o1_nums]
    o1_x = check_cols[o1_nums]
    o1_rng = np.arange(ref_pix, stop_1 - ref_pix)
    if len(o1_x) >= 2:
        o1_spline = UnivariateSpline(o1_x, o1_x0, k=5)
        o1_fit = o1_spline(o1_rng)
        fig.x(o1_x, o1_x0, color='magenta')
        fig.step(o1_rng, o1_fit, color='cyan')
    else:
        o1_fit = None

    # Fit polynomial to order 2 centers
    o2_x0 = np.array(ref_cols + [i['ord22_x0'] for i in results] + ref_cols)
    o2_nums = np.argwhere(~np.isnan(o2_x0)).flatten()
    o2_x0 = o2_x0[o2_nums]
    o2_x = check_cols[o2_nums]
    o2_rng = np.arange(ref_pix, stop_2)
    if len(o2_x) >= 2:
        o2_spline = UnivariateSpline(o2_x, o2_x0, k=5)
        o2_fit = o2_spline(o2_rng)
        fig.x(o2_x, o2_x0, color='magenta')
        fig.step(o2_rng, o2_fit, color='cyan')
    else:
        o2_fit = None

    # Fit polynomial to order 3 centers
    o3_x0 = np.array(ref_cols + [i['ord32_x0'] for i in results] + ref_cols)
    o3_nums = np.argwhere(~np.isnan(o3_x0)).flatten()
    o3_x0 = o3_x0[o3_nums]
    o3_x = check_cols[o3_nums]
    o3_rng = np.arange(ref_pix, stop_3)
    if len(o3_x) >= 2:
        o3_spline = UnivariateSpline(o3_x, o3_x0, k=5)
        o3_fit = o3_spline(o3_rng)
        fig.x(o3_x, o3_x0, color='magenta')
        fig.step(o3_rng, o3_fit, color='cyan')
    else:
        o3_fit = None

    if verbose:
        print('Trace finder finished: {} {}'.format(round(time.time() - start, 3), 's'))

    if plot:
        show(fig)

    return o1_fit, o2_fit, o3_fit, results


def fit_psfs(data, plot=False, order1=True, order2=True, order3=True, sigma=3.1, fwhm=12, **kwargs):

    """
    Function to fit PSFs in a column of SOSS data

    Parameters
    ----------
    data: sequence
        The column of data to fit
    plot: bool
        Plot the data and best fit

    Returns
    -------
    sequence
        The data masks for each order
    """
    # Global tolerances
    x0_tol = 0.5 # pixels
    amp_tol = 3 # factor
    std_tol = 1.5 # factor
    fwhm_tol = 1.5 # factor

    # Build model
    model = None
    names = []

    # =================================================================================================================
    # Order 1 Model ===================================================================================================
    # =================================================================================================================

    names1 = ['ord11_amp', 'ord11_mean', 'ord11_std',
              'ord12_amp', 'ord12_x0', 'ord12_fwhm',
              'ord13_amp', 'ord13_mean', 'ord13_std']

    if order1:

        # Order 1 params
        floor1, ceil1 = 37, 95
        sep1 = kwargs.get('sep1', 6.7)
        x012 = kwargs.get('ord12_x0', 70)
        mean11 = kwargs.get('ord11_mean', x012 - sep1)
        mean13 = kwargs.get('ord13_mean', x012 + sep1)
        std11 = kwargs.get('ord11_std', sigma)
        fwhm12 = kwargs.get('ord12_fwhm', fwhm)
        std13 = kwargs.get('ord13_std', sigma)
        amp11 = kwargs.get('ord11_amp', np.nanmax(data))
        amp12 = kwargs.get('ord12_amp', amp11 * 0.9)
        amp13 = kwargs.get('ord13_amp', amp11)

        # Set x0 bounds for order 1 center
        x012_min = max(floor1, kwargs.get('ord12_x0_min') or (x012 - x0_tol))
        x012_max = min(ceil1, kwargs.get('ord12_x0_max') or (x012 + x0_tol))

        # First Gaussian peak (mean_0, amplitude_0, stddev_0)
        o1g1_bounds = {'mean': (x012_min - sep1, x012_max), 'amplitude': (amp11 / amp_tol, amp11 * amp_tol), 'stddev': (std11 / std_tol, std11 * std_tol)}
        o1g1 = models.Gaussian1D(amplitude=amp11, mean=mean11, stddev=std11, bounds=o1g1_bounds)

        # Second Lorentzian peak (x_0_1, amplitude_1, fwhm_1)
        # o1g2_bounds = {'x_0': (x012_min, x012_max), 'amplitude': (min(amp12 / amp_tol, amp11), amp11 / amp_tol / 1.2), 'fwhm': (fwhm12 - fwhm_tol, fwhm12 + fwhm_tol)}
        o1g2_bounds = {'x_0': (x012_min, x012_max), 'amplitude': (min(amp12 / amp_tol, amp11), amp11), 'fwhm': (fwhm12 - fwhm_tol, fwhm12 + fwhm_tol)}
        o1g2 = models.Lorentz1D(amplitude=amp12, x_0=x012, fwhm=fwhm12, bounds=o1g2_bounds)

        # Third Gaussian peak (mean_2, amplitude_2, stddev_2)
        o1g3_bounds = {'mean': (x012_min, x012_max + sep1), 'amplitude': (amp13 / amp_tol, amp13 * amp_tol), 'stddev': (std13 / std_tol, std13 * std_tol)}
        o1g3 = models.Gaussian1D(amplitude=amp13, mean=mean13, stddev=std13, bounds=o1g3_bounds)

        # Tie the locations of the middle Lorentzian to halfway between the twin peaks
        def tie_center1(model):
            return (model.mean_0 + model.mean_2) / 2.
        o1g2.x_0.tied = tie_center1

        # Tie first peak to center peak - separation
        def tie_sep_o1g1(model):
            return model.x_0_1 - sep1
        o1g1.mean.tied = tie_sep_o1g1

        # Tie third peak to center peak + separation
        def tie_sep_o1g3(model):
            return model.x_0_1 + sep1
        o1g3.mean.tied = tie_sep_o1g3

        # Tie the peak stddev
        def tie_fwhm_o1g1(model):
            return model.stddev_0
        o1g3.stddev.tied = tie_fwhm_o1g1

        # Tie the peak stddev
        def tie_fwhm_o1g3(model):
            return model.stddev_2
        o1g1.stddev.tied = tie_fwhm_o1g3

        # Composite model
        if model is None:
            model = o1g1 + o1g2 + o1g3
        else:
            model += o1g1 + o1g2 + o1g3
        names += names1

    # =================================================================================================================
    # Order 2 Model ===================================================================================================
    # =================================================================================================================

    names2 = ['ord21_amp', 'ord21_mean', 'ord21_std',
              'ord22_amp', 'ord22_x0', 'ord22_fwhm',
              'ord23_amp', 'ord23_mean', 'ord23_std']

    if order2:

        # Order 2 params
        floor2, ceil2 = 90, 256
        sep2 = kwargs.get('sep2', 6.5)
        x022 = kwargs.get('ord22_x0', 100)
        mean21 = kwargs.get('ord21_mean', x022 - sep2)
        mean23 = kwargs.get('ord23_mean', x022 + sep2)
        std21 = kwargs.get('ord21_std', sigma)
        fwhm22 = kwargs.get('ord22_fwhm', fwhm)
        std23 = kwargs.get('ord23_std', sigma)
        amp21 = kwargs.get('ord21_amp', np.nanmax(data[105:ceil2]))
        amp22 = kwargs.get('ord22_amp', amp21 * 0.8)
        amp23 = kwargs.get('ord23_amp', amp21)

        # Set x0 bounds for order 2 center
        x022_min = max(floor2, kwargs.get('ord22_x0_min') or (x022 - x0_tol))
        x022_max = min(ceil2, kwargs.get('ord22_x0_max') or (x022 + x0_tol))

        # First Gaussian peak (mean_3, amplitude_3, stddev_3)
        o2g1_bounds = {'mean': (x022_min - sep2, x022_max), 'amplitude': (amp21 / amp_tol, amp21 * amp_tol), 'stddev': (std21 / std_tol, std21 * std_tol)}
        o2g1 = models.Gaussian1D(amplitude=amp21, mean=mean21, stddev=std21, bounds=o2g1_bounds)

        # Second Lorentzian peak (x_0_4, amplitude_4, fwhm_4)
        o2g2_bounds = {'x_0': (x022_min, x022_max), 'amplitude':  (min(amp22 / amp_tol, amp22), amp22 / amp_tol / 1.2), 'fwhm': (fwhm22 / fwhm_tol, fwhm22 * fwhm_tol)}
        o2g2 = models.Lorentz1D(amplitude=amp22, x_0=x022, fwhm=fwhm22, bounds=o2g2_bounds)

        # Third Gaussian peak (mean_5, amplitude_5, stddev_5)
        o2g3_bounds = {'mean': (x022_min, x022_max + sep2), 'amplitude': (amp23 / amp_tol, amp23 * amp_tol), 'stddev': (std23 / std_tol, std23 * std_tol)}
        o2g3 = models.Gaussian1D(amplitude=amp23, mean=mean23, stddev=std23, bounds=o2g3_bounds)

        # Tie the locations of the middle Lorentzian to halfway between the twin peaks
        def tie_center2(model):
            return (model.mean_3 + model.mean_5) / 2.
        o2g2.x_0.tied = tie_center2

        # Tie first peak to center peak - separation
        def tie_sep_o2g1(model):
            attr = getattr(model, 'x_0_{}'.format(4 if order1 else 1))
            return attr - sep2
        o2g1.mean.tied = tie_sep_o2g1

        # Tie third peak to center peak + separation
        def tie_sep_o2g3(model):
            attr = getattr(model, 'x_0_{}'.format(4 if order1 else 1))
            return attr + sep2
        o2g3.mean.tied = tie_sep_o2g3

        # Tie the peak stddev
        def tie_fwhm_o2g1(model):
            attr = getattr(model, 'stddev_{}'.format(3 if order1 else 0))
            return attr
        o2g3.stddev.tied = tie_fwhm_o2g1

        # Tie the peak stddev
        def tie_fwhm_o2g3(model):
            attr = getattr(model, 'stddev_{}'.format(5 if order1 else 2))
            return attr
        o2g1.stddev.tied = tie_fwhm_o2g3

        # Composite model
        if model is None:
            model = o2g1 + o2g2 + o2g3
        else:
            model += o2g1 + o2g2 + o2g3
        names += names2

    # =================================================================================================================
    # Order 3 Model ===================================================================================================
    # =================================================================================================================

    names3 = ['ord31_amp', 'ord31_mean', 'ord31_std',
              'ord32_amp', 'ord32_x0', 'ord32_fwhm',
              'ord33_amp', 'ord33_mean', 'ord33_std']

    if order3:

        # Order 3 params
        floor3, ceil3 = 140, 210
        sep3 = kwargs.get('sep3', 7)
        x032 = kwargs.get('ord32_x0', 198)
        mean31 = kwargs.get('ord31_mean', x032 - sep3)
        mean33 = kwargs.get('ord33_mean', x032 + sep3)
        std31 = kwargs.get('ord31_std', sigma)
        fwhm32 = kwargs.get('ord32_fwhm', fwhm)
        std33 = kwargs.get('ord33_std', sigma)
        amp31 = kwargs.get('ord31_amp', np.nanmax(data[floor3:ceil3]))
        amp32 = kwargs.get('ord32_amp', amp31 * 0.8)
        amp33 = kwargs.get('ord33_amp', amp31)

        # Set x0 bounds for order 3 center
        x032_min = max(floor3, kwargs.get('ord32_x0_min') or (x032 - x0_tol))
        x032_max = min(ceil3, kwargs.get('ord32_x0_max') or (x032 + x0_tol))

        # First Gaussian peak (mean_6, amplitude_6, stddev_6)
        o3g1_bounds = {'mean': (x032_min - sep3, x032_max), 'amplitude': (amp31 / amp_tol, amp31 * amp_tol), 'stddev': (std31 / std_tol, std31 * std_tol)}
        o3g1 = models.Gaussian1D(amplitude=amp31, mean=mean31, stddev=std31, bounds=o3g1_bounds)

        # Second Lorentzian peak (x_0_7, amplitude_7, fwhm_7)
        o3g2_bounds = {'x_0': (x032_min, x032_max), 'amplitude':  (min(amp32 / amp_tol, amp32), amp32 / amp_tol / 1.2), 'fwhm': (fwhm32 / fwhm_tol, fwhm32 * fwhm_tol)}
        o3g2 = models.Lorentz1D(amplitude=amp32, x_0=x032, fwhm=fwhm32, bounds=o3g2_bounds)

        # Third Gaussian peak (mean_8, amplitude_8, stddev_8)
        o3g3_bounds = {'mean': (x032_min, x032_max + sep3), 'amplitude': (amp33 / amp_tol, amp33 * amp_tol), 'stddev': (std33 / std_tol, std33 * std_tol)}
        o3g3 = models.Gaussian1D(amplitude=amp33, mean=mean33, stddev=std33, bounds=o3g3_bounds)

        # Tie the locations of the middle Lorentzian to halfway between the twin peaks
        def tie_center3(model):
            return (model.mean_6 + model.mean_8) / 2.
        o3g2.x_0.tied = tie_center3

        # Tie first peak to center peak - separation
        def tie_sep_o3g1(model):
            attr = getattr(model, 'x_0_{}'.format(1 + (3 if order1 else 0) + (3 if order2 else 0)))
            return attr - sep3
        o3g1.mean.tied = tie_sep_o3g1

        # Tie third peak to center peak + separation
        def tie_sep_o3g3(model):
            attr = getattr(model, 'x_0_{}'.format(1 + (3 if order1 else 0) + (3 if order2 else 0)))
            return attr + sep3
        o3g3.mean.tied = tie_sep_o3g3

        # Tie the peak stddev
        def tie_fwhm_o3g1(model):
            attr = getattr(model, 'stddev_{}'.format(0 + (3 if order1 else 0) + (3 if order2 else 0)))
            return attr
        o3g3.stddev.tied = tie_fwhm_o3g1

        # Tie the peak stddev
        def tie_fwhm_o3g3(model):
            attr = getattr(model, 'stddev_{}'.format(2 + (3 if order1 else 0) + (3 if order2 else 0)))
            return attr
        o3g1.stddev.tied = tie_fwhm_o3g3

        # Composite model
        if model is None:
            model = o3g1 + o3g2 + o3g3
        else:
            model += o3g1 + o3g2 + o3g3
        names += names3

    # =================================================================================================================

    # Initialize fitter
    fit_model = fitting.LevMarLSQFitter()
    x = np.arange(data.size)

    # Fit the composite model
    soln = fit_model(model, x, data)

    # Put solution in dictionary
    params = {k: v for k, v in zip(names, soln.parameters)}
    if order1:
        params['sep1'] = sep1
    else:
        for p in names1:
            params[p] = np.nan
    if order2:
        params['sep2'] = sep2
    else:
        for p in names2:
            params[p] = np.nan
    if order3:
        params['sep3'] = sep3
    else:
        for p in names3:
            params[p] = np.nan

    # Determine fit quality
    func_form = functional_psf(params)
    params['residuals'] = func_form - data
    print('{} / {} counts ({}%) captured'.format(int(np.nansum(func_form)), int(np.nansum(data)), int(np.nansum(func_form)*100/np.nansum(data))))

    if plot:
        fit_plot(data, params)

    return params


def functional_psf(params, order=None):
    """
    Generate the PSF with the given parameters

    Parameters
    ----------
    params: dict
        The dictionary of the parameters to evaluate

    Returns
    -------
    sequence
        The functional form of the PSF
    """
    x = np.arange(256)
    y11 = params['ord11_amp'] * np.exp(-0.5 * (x - params['ord11_mean']) ** 2 / params['ord11_std'] ** 2)
    y12 = (params['ord12_amp'] * (params['ord12_fwhm'] / 2)**2) / ((params['ord12_fwhm'] / 2)**2 + (x - params['ord12_x0'])**2)
    y13 = params['ord13_amp'] * np.exp(-0.5 * (x - params['ord13_mean']) ** 2 / params['ord13_std'] ** 2)
    y21 = params['ord21_amp'] * np.exp(-0.5 * (x - params['ord21_mean']) ** 2 / params['ord21_std'] ** 2)
    y22 = (params['ord22_amp'] * (params['ord22_fwhm'] / 2)**2) / ((params['ord22_fwhm'] / 2)**2 + (x - params['ord22_x0'])**2)
    y23 = params['ord23_amp'] * np.exp(-0.5 * (x - params['ord23_mean']) ** 2 / params['ord23_std'] ** 2)
    y31 = params['ord31_amp'] * np.exp(-0.5 * (x - params['ord31_mean']) ** 2 / params['ord31_std'] ** 2)
    y32 = (params['ord32_amp'] * (params['ord32_fwhm'] / 2)**2) / ((params['ord32_fwhm'] / 2)**2 + (x - params['ord32_x0'])**2)
    y33 = params['ord33_amp'] * np.exp(-0.5 * (x - params['ord33_mean']) ** 2 / params['ord33_std'] ** 2)

    # No order 1
    if np.isnan(params['ord12_x0']):
        y11 = y12 = y13 = np.zeros_like(x)

    # No order 2
    if np.isnan(params['ord22_x0']):
        y21 = y22 = y23 = np.zeros_like(x)

    # No order 3
    if np.isnan(params['ord32_x0']):
        y31 = y32 = y33 = np.zeros_like(x)

    if order == 1:
        return y11, y12, y13
    if order == 2:
        return y21, y22, y23
    if order == 3:
        return y31, y32, y33
    else:
        return y11 + y12 + y13 + y21 + y22 + y23 + y31 + y32 + y33


def fit_plot(data, params, order1=None, order2=None, order3=None):
    """
    Create interactive plot of the given parameters

    Parameters
    ----------
    data: sequence
        The data to plot
    """
    # Check for order 1
    order1 = order1 or not np.isnan(params['ord12_x0'])

    # Check for order 2
    order2 = order2 or not np.isnan(params['ord22_x0'])

    # Check for order 3
    order3 = order3 or not np.isnan(params['ord32_x0'])

    # Evaluate solution
    x = np.arange(data.size)
    if order1:
        y11, y12, y13 = functional_psf(params, order=1)
    else:
        y11 = y12 = y13 = np.zeros(256)
    if order2:
        y21, y22, y23 = functional_psf(params, order=2)
    else:
        y21 = y22 = y23 = np.zeros(256)
    if order3:
        y31, y32, y33 = functional_psf(params, order=3)
    else:
        y31 = y32 = y33 = np.zeros(256)

    # Add it up
    y = y11 + y12 + y13 + y21 + y22 + y23 + y31 + y32 + y33

    source = ColumnDataSource(data=dict(x=x, y=y, y11=y11, y12=y12, y13=y13, y21=y21, y22=y22, y23=y23, y31=y31, y32=y32, y33=y33, raw=data, res=data - y))
    plot = figure(x_range=(0, 256), plot_width=600, plot_height=500)
    plot.circle('x', 'raw', source=source)
    plot.line('x', 'y', source=source, line_color='black', line_width=2, line_alpha=0.6)
    plot.line('x', 'y11', source=source, line_color='blue', line_width=1, line_alpha=0.6)
    plot.line('x', 'y12', source=source, line_color='red', line_width=1, line_alpha=0.6)
    plot.line('x', 'y13', source=source, line_color='green', line_width=1, line_alpha=0.6)
    plot.line('x', 'y21', source=source, line_color='blue', line_width=1, line_alpha=0.6)
    plot.line('x', 'y22', source=source, line_color='red', line_width=1, line_alpha=0.6)
    plot.line('x', 'y23', source=source, line_color='green', line_width=1, line_alpha=0.6)
    plot.line('x', 'y31', source=source, line_color='blue', line_width=1, line_alpha=0.6)
    plot.line('x', 'y32', source=source, line_color='red', line_width=1, line_alpha=0.6)
    plot.line('x', 'y33', source=source, line_color='green', line_width=1, line_alpha=0.6)
    # plot.x_range = Range1d(params['ord12_x0'] - 50, params['ord12_x0'] + 50)
    # plot.y_range = Range1d(0, params['ord11_amp'] * 1.5)
    crosshair = CrosshairTool(dimensions="height")
    plot.add_tools(crosshair)

    # Plot residuals in each column
    res_plot = figure(x_range=plot.x_range, plot_width=600, plot_height=200)
    res_plot.line('x', 'res', source=source, color='red')
    res_plot.line(x, np.zeros_like(x), color='black')
    res_plot.add_tools(crosshair)

    # Order 1 sliders
    if order1:
        amp_slider_1 = Slider(start=0, end=params['ord12_amp'] * 2, value=params['ord12_amp'], step=1., title="Amp")
        x0_slider_1 = Slider(start=0, end=256, value=params['ord12_x0'], step=.1, title="x0")
        fwhm_slider_1 = Slider(start=0, end=params['ord12_fwhm'] * 2, value=params['ord12_fwhm'], step=.1, title="FWHM")
        sep_slider_1 = Slider(start=0, end=15, value=params['sep1'], step=.1, title="Separation")
        twin_amp_slider_1 = Slider(start=0, end=params['ord11_amp'] * 2, value=params['ord11_amp'], step=.1, title="Twin Amp")
        twin_std_slider_1 = Slider(start=0, end=params['ord11_std'] * 2, value=params['ord11_std'], step=.1, title="Twin Sigma")

    # Order 2 sliders
    if order2:
        amp_slider_2 = Slider(start=0, end=params['ord22_amp'] * 2, value=params['ord22_amp'], step=1., title="Amp")
        x0_slider_2 = Slider(start=0, end=256, value=params['ord22_x0'], step=.1, title="x0")
        fwhm_slider_2 = Slider(start=0, end=params['ord22_fwhm'] * 2, value=params['ord22_fwhm'], step=.1, title="FWHM")
        sep_slider_2 = Slider(start=0, end=15, value=params['sep2'], step=.1, title="Separation")
        twin_amp_slider_2 = Slider(start=0, end=params['ord21_amp'] * 2, value=params['ord21_amp'], step=.1, title="Twin Amp")
        twin_std_slider_2 = Slider(start=0, end=params['ord21_std'] * 2, value=params['ord21_std'], step=.1, title="Twin Sigma")
        
    # Order 3 sliders
    if order3:
        amp_slider_3 = Slider(start=0, end=params['ord32_amp'] * 2, value=params['ord32_amp'], step=1., title="Amp")
        x0_slider_3 = Slider(start=0, end=256, value=params['ord32_x0'], step=.1, title="x0")
        fwhm_slider_3 = Slider(start=0, end=params['ord32_fwhm'] * 2, value=params['ord32_fwhm'], step=.1, title="FWHM")
        sep_slider_3 = Slider(start=0, end=15, value=params['sep3'], step=.1, title="Separation")
        twin_amp_slider_3 = Slider(start=0, end=params['ord31_amp'] * 2, value=params['ord31_amp'], step=.1, title="Twin Amp")
        twin_std_slider_3 = Slider(start=0, end=params['ord31_std'] * 2, value=params['ord31_std'], step=.1, title="Twin Sigma")

    # Build Javascript code
    code ="""
    const data = source.data;
    const x = data['x']
    const y = data['y']
    const raw = data['raw']
    const res = data['res']
    """

    if order1:
        code += """
    const amp_val_1 = amp_1.value;
    const x0_val_1 = x0_1.value;
    const fwhm_val_1 = fwhm_1.value;
    const sep_val_1 = sep_1.value;
    const twin_amp_val_1 = twin_amp_1.value;
    const twin_std_val_1 = twin_std_1.value;
    const y11 = data['y11']
    const y12 = data['y12']
    const y13 = data['y13']
    """
    
    if order2:
        code += """
    const amp_val_2 = amp_2.value;
    const x0_val_2 = x0_2.value;
    const fwhm_val_2 = fwhm_2.value;
    const sep_val_2 = sep_2.value;
    const twin_amp_val_2 = twin_amp_2.value;
    const twin_std_val_2 = twin_std_2.value;
    const y21 = data['y21']
    const y22 = data['y22']
    const y23 = data['y23']
    """
        
    if order3:
        code += """
    const amp_val_3 = amp_3.value;
    const x0_val_3 = x0_3.value;
    const fwhm_val_3 = fwhm_3.value;
    const sep_val_3 = sep_3.value;
    const twin_amp_val_3 = twin_amp_3.value;
    const twin_std_val_3 = twin_std_3.value;
    const y31 = data['y31']
    const y32 = data['y32']
    const y33 = data['y33']
    """

    code += """
    for (var i = 0; i < x.length; i++) {
    """

    if order1:
        code += """
        y11[i] = twin_amp_val_1 * Math.exp(-0.5 * (x[i] - x0_val_1 + sep_val_1) ** 2 / twin_std_val_1 ** 2);
        y12[i] = (amp_val_1 * Math.pow(fwhm_val_1 / 2, 2)) / (Math.pow(fwhm_val_1 / 2, 2) + Math.pow(x[i] - x0_val_1, 2));
        y13[i] = twin_amp_val_1 * Math.exp(-0.5 * (x[i] - x0_val_1 - sep_val_1) ** 2 / twin_std_val_1 ** 2);
    """

    if order2:
        code += """
        y21[i] = twin_amp_val_2 * Math.exp(-0.5 * (x[i] - x0_val_2 + sep_val_2) ** 2 / twin_std_val_2 ** 2);
        y22[i] = (amp_val_2 * Math.pow(fwhm_val_2 / 2, 2)) / (Math.pow(fwhm_val_2 / 2, 2) + Math.pow(x[i] - x0_val_2, 2));
        y23[i] = twin_amp_val_2 * Math.exp(-0.5 * (x[i] - x0_val_2 - sep_val_2) ** 2 / twin_std_val_2 ** 2);
    """

    if order3:
        code += """
        y31[i] = twin_amp_val_3 * Math.exp(-0.5 * (x[i] - x0_val_3 + sep_val_3) ** 2 / twin_std_val_3 ** 2);
        y32[i] = (amp_val_3 * Math.pow(fwhm_val_3 / 2, 2)) / (Math.pow(fwhm_val_3 / 2, 2) + Math.pow(x[i] - x0_val_3, 2));
        y33[i] = twin_amp_val_3 * Math.exp(-0.5 * (x[i] - x0_val_3 - sep_val_3) ** 2 / twin_std_val_3 ** 2);
    """

    code += "    y[i] = {}{}{}{}{};".format('y11[i] + y12[i] + y13[i]' if order1 else '', '+' if order1 else '',
                                            'y21[i] + y22[i] + y23[i]' if order2 else '', '+' if order1 or order2 else '',
                                            'y31[i] + y32[i] + y33[i]' if order3 else '')

    code += """
        res[i] = raw[i] - y[i];
    }
    source.change.emit();
    """

    # Build Javascript args
    js_args = {'source': source}
    if order1:
        js_args.update({'amp_1': amp_slider_1, 'x0_1': x0_slider_1, 'fwhm_1': fwhm_slider_1, 'sep_1': sep_slider_1,
                        'twin_amp_1': twin_amp_slider_1, 'twin_std_1': twin_std_slider_1})
    if order2:
        js_args.update({'amp_2': amp_slider_2, 'x0_2': x0_slider_2, 'fwhm_2': fwhm_slider_2, 'sep_2': sep_slider_2,
                        'twin_amp_2': twin_amp_slider_2, 'twin_std_2': twin_std_slider_2})
    if order3:
        js_args.update({'amp_3': amp_slider_3, 'x0_3': x0_slider_3, 'fwhm_3': fwhm_slider_3, 'sep_3': sep_slider_3,
                        'twin_amp_3': twin_amp_slider_3, 'twin_std_3': twin_std_slider_3})

    # JS callback
    callback = CustomJS(args=js_args, code=code)
    if order1:
        amp_slider_1.js_on_change('value', callback)
        x0_slider_1.js_on_change('value', callback)
        fwhm_slider_1.js_on_change('value', callback)
        sep_slider_1.js_on_change('value', callback)
        twin_amp_slider_1.js_on_change('value', callback)
        twin_std_slider_1.js_on_change('value', callback)
    if order2:
        amp_slider_2.js_on_change('value', callback)
        x0_slider_2.js_on_change('value', callback)
        fwhm_slider_2.js_on_change('value', callback)
        sep_slider_2.js_on_change('value', callback)
        twin_amp_slider_2.js_on_change('value', callback)
        twin_std_slider_2.js_on_change('value', callback)
    if order3:
        amp_slider_3.js_on_change('value', callback)
        x0_slider_3.js_on_change('value', callback)
        fwhm_slider_3.js_on_change('value', callback)
        sep_slider_3.js_on_change('value', callback)
        twin_amp_slider_3.js_on_change('value', callback)
        twin_std_slider_3.js_on_change('value', callback)

    slider_col = None
    if order1:
        if slider_col is None:
            slider_col = [Div(text="Order 1"), x0_slider_1, amp_slider_1, fwhm_slider_1, sep_slider_1, twin_amp_slider_1, twin_std_slider_1]
        else:
            slider_col += [Div(text="Order 1"), x0_slider_1, amp_slider_1, fwhm_slider_1, sep_slider_1, twin_amp_slider_1, twin_std_slider_1]
    if order2:
        if slider_col is None:
            slider_col = [Div(text="Order 2"), x0_slider_2, amp_slider_2, fwhm_slider_2, sep_slider_2, twin_amp_slider_2, twin_std_slider_2]
        else:
            slider_col += [Div(text="Order 2"), x0_slider_2, amp_slider_2, fwhm_slider_2, sep_slider_2, twin_amp_slider_2, twin_std_slider_2]
    if order3:
        if slider_col is None:
            slider_col = [Div(text="Order 3"), x0_slider_3, amp_slider_3, fwhm_slider_3, sep_slider_3, twin_amp_slider_3, twin_std_slider_3]
        else:
            slider_col += [Div(text="Order 3"), x0_slider_3, amp_slider_3, fwhm_slider_3, sep_slider_3, twin_amp_slider_3, twin_std_slider_3]

    # Show plot
    layout = row(column(plot, res_plot), column(*slider_col))
    show(layout)

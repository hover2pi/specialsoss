# -*- coding: utf-8 -*-

"""A module of shared tools for SOSS data"""

from copy import copy
import os
from pkg_resources import resource_filename

from astropy.io import fits
from bokeh.plotting import figure, show
from hotsoss.plotting import plot_frame
from hotsoss import utils
from jwst import datamodels as dm
import numpy as np


def get_references(subarray, filter='CLEAR', context='jwst_niriss_0134.imap'):
    """
    Get dictionary of the reference file locations for the given subarray

    Parameters
    ----------
    subarray: str
        The subarray to use, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
    filter: str
        The filter to use, ['CLEAR', 'F277W']

    Returns
    -------
    dict
        The dictionary of reference files
    """
    # Accepted subarrays
    subarrays = ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
    if subarray not in subarrays:
        raise ValueError("{} is not a supported subarray. Please use {}".format(subarray, subarrays))

    # Accepted filters
    filters = ['CLEAR', 'F277W']
    if filter not in filters:
        raise ValueError("{} is not a supported filter. Please use {}".format(filter, filters))

    # TODO: F277W not yet supported. Just delete this line when F277W support is added to crds
    filter = 'CLEAR'

    params = {"INSTRUME": "NIRISS",
              "READPATT": "NIS",
              "EXP_TYPE": "NIS_SOSS",
              "DETECTOR": "NIS",
              "PUPIL": "GR700XD",
              "DATE-OBS": "2020-07-28",
              "TIME-OBS": "00:00:00",
              "INSTRUMENT": "NIRISS",
              "FILTER": filter,
              "SUBARRAY": subarray}

    # Default ref file path
    default_path = resource_filename('specialsoss', 'files/refs/')

    # Collect reference files for subarray+filter combination
    try:
        import crds
        refs = crds.getreferences(params, context=context)
    except:

        refs = {'saturation': os.path.join(default_path, 'jwst_niriss_saturation_0010.fits'),
                'photom': os.path.join(default_path, 'jwst_niriss_photom_0037.fits'),
                'flat': os.path.join(default_path, 'jwst_niriss_flat_0190.fits'),
                'gain': os.path.join(default_path, 'jwst_niriss_gain_0005.fits'),
                'superbias': os.path.join(default_path, 'jwst_niriss_superbias_0120.fits'),
                'dark': os.path.join(default_path, 'jwst_niriss_dark_0114.fits'),
                'readnoise': os.path.join(default_path, 'jwst_niriss_readnoise_0001.fits'),
                'linearity': os.path.join(default_path, 'jwst_niriss_linearity_0011.fits')}

        if subarray == 'SUBSTRIP96':
            refs['superbias'] = os.path.join(default_path, 'jwst_niriss_superbias_0111.fits')
            refs['dark'] = os.path.join(default_path, 'jwst_niriss_dark_0111.fits')

        if subarray == 'FULL':
            refs['gain'] = os.path.join(default_path, 'jwst_niriss_gain_0002.fits')
            refs['superbias'] = os.path.join(default_path, 'jwst_niriss_superbias_0029.fits')
            refs['dark'] = os.path.join(default_path, 'jwst_niriss_dark_0129.fits')

    # Check if reference files exist and load defaults if necessary
    for ref_name, ref_fn in refs.items():
        if not 'NOT FOUND' in ref_fn and not os.path.isfile(refs[ref_name]):
            refs[ref_name] = os.path.join(default_path, ref_fn)
            print("Could not get {} reference file from CRDS. Using {}.".format(ref_name, ref_fn))

    return refs


def convert_cv3(filepath, filename=None, filter='CLEAR', subarray='SUBSTRIP256'):
    """
    Convert a CV3 file to a pipeline readable '_uncal' file

    Parameters
    ----------
    filepath: str
        The path to the CV3 file

    Returns
    -------
    str
        The new filepath
    """
    # Open the CV3 file
    with fits.open(filepath) as hdul:
        hdul.verify('fix')
        data = hdul[0].data
        head = hdul[0].header
        total_frames, ncols, nrows = data.shape
        nints = head['NINT']
        ngrps = head['NGROUP']
        date, time = head['DATE'].split('T')
        time += '.00'
        new_shape = nints, ngrps, nrows, ncols
        data = data.swapaxes(1, 2).reshape(new_shape)[:, :, ::-1, ::-1]

    # Get subarray specifics
    groupgap = 0
    nframes = nsample = 1
    nresets = nresets1 = nresets2 = 1
    dropframes1 = dropframes3 = 0
    pix = utils.subarray_specs(subarray)
    tframe = pix.get('tfrm')
    exposure_time = tframe * ((ngrps * total_frames + (ngrps - 1) * groupgap + dropframes1) * nints)
    duration = exposure_time + tframe * (dropframes3 * nints + nresets1 + nresets2 * (nints - 1))

    # Make a RampModel
    groupdq = np.zeros_like(data)
    pixeldq = np.zeros((nrows, ncols))
    err = np.random.normal(loc=data, scale=data / 100.)
    mod = dm.RampModel(data=data, groupdq=groupdq, pixeldq=pixeldq, err=err)

    # Set meta data values for header keywords
    mod.meta.telescope = 'JWST'
    mod.meta.instrument.name = 'NIRISS'
    mod.meta.instrument.detector = 'NIS'
    mod.meta.instrument.filter = filter
    mod.meta.instrument.pupil = 'GR700XD'
    mod.meta.exposure.type = 'NIS_SOSS'
    mod.meta.exposure.nints = nints
    mod.meta.exposure.ngroups = ngrps
    mod.meta.exposure.nframes = nframes
    mod.meta.exposure.readpatt = 'NISRAPID'
    mod.meta.exposure.groupgap = 0
    mod.meta.exposure.frame_time = tframe
    mod.meta.exposure.group_time = nframes * tframe
    mod.meta.exposure.exposure_time = exposure_time
    mod.meta.exposure.duration = duration
    mod.meta.exposure.nresets_at_start = 1
    mod.meta.exposure.nresets_between_ints = 1
    mod.meta.subarray.name = subarray
    mod.meta.subarray.xsize = ncols
    mod.meta.subarray.ysize = nrows
    mod.meta.subarray.xstart = pix.get('xloc', 1)
    mod.meta.subarray.ystart = pix.get('yloc', 1)
    mod.meta.subarray.fastaxis = -2
    mod.meta.subarray.slowaxis = -1
    mod.meta.observation.date = date
    mod.meta.observation.time = time
    mod.meta.target.ra = 1.23
    mod.meta.target.dec = 2.34
    mod.meta.target.source_type = 'POINT'

    # Save the file
    outfile = resource_filename('specialsoss', filename or 'files/CV3_{}_{}_uncal.fits'.format(subarray, filter))
    mod.save(outfile, overwrite=True)


def bin_counts(data, wavebins, pixel_mask=None, plot_bin=None):
    """
    Bin the counts in data given the wavelength bin information

    Parameters
    ----------
    data: array-like
        The 3D or 4D data to bin
    wavebins: sequence
        A list of lists of the pixels in each wavelength bin
    pixel_mask: array-like (optional)
        A 2D mask of 1s and 0s to apply to the data
    plot_bin: int
        The index of the bin to plot

    Returns
    -------
    np.ndarray
        The counts in each wavelength bin for each frame in data
    """
    # Reshape into 3D
    data, dims = to_3d(data)

    # Array to store counts
    counts = np.zeros((data.shape[0], len(wavebins)), dtype=float)

    # Apply the pixel mask by multiplying non-signal pixels by 0 before adding
    if isinstance(pixel_mask, np.ndarray) and pixel_mask.shape == data.shape[-2:]:
        data *= pixel_mask[None, :, :]

    # Add up the counts in each bin in each frame
    for n, (xpix, ypix) in enumerate(wavebins):
        cutoff = xpix < dims[-2]
        counts[:, n] = np.nansum(data[:, xpix[cutoff], ypix[cutoff]], axis=1)

    # Plot a bin for visual inspection
    if isinstance(plot_bin, int):

        # Make frame from binned pixels
        binpix = wavebins[plot_bin]
        binframe = copy(data[-1])
        binframe[binpix[0], binpix[1]] *= 10
        fig = plot_frame(binframe, title="Bin {}: {:.0f} counts in {:.2f} pixels".format(plot_bin, counts[-1][plot_bin], np.nansum(pixel_mask[binpix[0], binpix[1]])))
        show(fig)

    # Restore original shape
    counts = counts.reshape((*list(dims[:-2]), counts.shape[-1]))

    return counts


def combine_spectra(s1, s2):
    """
    Make a composite spectrum from two spectra

    Parameters
    ----------
    s1: sequence
        The [W, F, E] of the first spectrum
    s2: sequence
        The [W, F, E] of the second spectrum

    Returns
    -------
    sequence
        The [W, F, E] of the combined spectrum
    """
    # Remove NaN and zero wavelengths
    idx1 = np.where(s1[0] > 0.)[0]
    idx2 = np.where(s2[0] > 0.)[0]
    s1 = np.array([i[idx1] for i in s1])
    s2 = np.array([i[idx2] for i in s2])

    # Determine if overlapping
    overlap = True
    try:
        if s1[0][-1] > s1[0][0] > s2[0][-1] or s2[0][-1] > s2[0][0] > s1[0][-1]:
            overlap = False
    except IndexError:
        overlap = False

    # Concatenate and order two segments if no overlap
    if not overlap:

        # Drop uncertainties on both spectra if one is missing
        if s1[2] is None or s2[2] is None:
            s1 = s1[:2]
            s2 = s2[:2]

        # Concatenate arrays and sort by wavelength
        new_spec = np.concatenate([s1, s2], axis=1).T
        new_spec = new_spec[np.argsort(new_spec[:, 0])].T

    # Otherwise there are three segments, (left, overlap, right)
    else:

        # Get the left segemnt
        left = s1[:, s1[0] <= s2[0][0]]
        if not np.any(left):
            left = s2[:, s2[0] <= s1[0][0]]

        # Get the right segment
        right = s1[:, s1[0] >= s2[0][-1]]
        if not np.any(right):
            right = s2[:, s2[0] >= s1[0][-1]]

        # Get the overlapping segements
        o1 = s1[:, np.where((s1[0] < right[0][0]) & (s1[0] > left[0][-1]))].squeeze()
        o2 = s2[:, np.where((s2[0] < right[0][0]) & (s2[0] > left[0][-1]))].squeeze()

        # Get the resolutions
        r1 = s1.shape[1]/(max(s1[0])-min(s1[0]))
        r2 = s2.shape[1]/(max(s2[0])-min(s2[0]))

        # Make higher resolution s1
        if r1 < r2:
            o1, o2 = o2, o1

        # Interpolate s2 to s1
        o2_flux = np.interp(o1[0], s2[0], s2[1])

        # Get the average
        o_flux = np.nanmean([o1[1], o2_flux], axis=0)

        # Calculate uncertainties if possible
        if len(s2) == len(o1) == 3:
            o2_unc = np.interp(o1[0], s2[0], s2[2])
            o_unc = np.sqrt(o1[2]**2 + o2_unc**2)
            overlap = np.array([o1[0], o_flux, o_unc])
        else:
            overlap = np.array([o1[0], o_flux])
            left = left[:2]
            right = right[:2]

        # Make sure it is 2D
        if overlap.shape == (3,):
            overlap.shape = 3, 1
        if overlap.shape == (2,):
            overlap.shape = 2, 1

        # Concatenate the segments
        new_spec = np.concatenate([left, overlap, right], axis=1)

    return new_spec


def nan_reference_pixels(data):
    """
    Convert reference pixels in SOSS data to NaN values
    """
    # Convert to 3D
    data, dims = to_3d(data)

    # Left, right (all subarrays)
    if dims[-1] == 2048:
        data[:, :, :4] = np.nan
        data[:, :, -4:] = np.nan

    # Top (excluding SUBSTRIP96)
    if dims[-2] in [256, 2048]:
        data[:, -4:, :] = np.nan

    # Bottom (Only FULL frame)
    if dims[-2] == 2048:
        data[:, :4, :] = np.nan

    # Return to original shape
    data = data.reshape(dims)

    return data


def test_simulations():
    """
    Make test simulations using awesimsoss
    """
    try:
        from awesimsoss import ModelTSO

        # Save location
        path = resource_filename('specialsoss', 'files/')

        # Delete old files
        os.system('rm {}'.format(os.path.join(path, '*.fits')))

        for filt in ['CLEAR', 'F277W']:
            for subarr in ['SUBSTRIP96', 'SUBSTRIP256']:

                tso = ModelTSO(jmag=8, nints=2, ngrps=2, subarray=subarr, filter=filt)
                tso.export(os.path.join(path, '{}_{}_uncal.fits'.format(subarr, filt)))

    except ImportError:
        print("Please install awesimsoss to generate test simulations.")


def to_3d(data):
    """
    Class to convert arbitrary data into 3 dimensions

    Parameters
    ----------
    data: array-like
        The data to reshape

    Returns
    -------
    new_data, old_shape
        The reshaped data and the old shape
    """
    # Ensure it's an array
    if isinstance(data, list):
        data = np.asarray(data)

    # Get the data shape
    old_shape = data.shape

    # Convert to 3D
    if data.ndim == 4:
        new_data = data.reshape((old_shape[0]*old_shape[1], old_shape[2], old_shape[3]))
    elif data.ndim == 3:
        new_data = copy(data)
    elif data.ndim == 2:
        new_data = data[None, :, :]
    else:
        raise ValueError("{}: Data must be in 2, 3, or 4 dimensions.".format(old_shape))

    return new_data, old_shape

#!/usr/bin/env python
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from jwst import datamodels
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from matplotlib.colors import LogNorm
from shutil import copyfile
from jwst.pipeline.calwebb_sloper import SloperPipeline
from jwst.pipeline import Spec2Pipeline
from jwst.linearity import LinearityStep
from jwst.dq_init import DQInitStep
from jwst.saturation import SaturationStep
from jwst.ipc import IPCStep
from jwst.superbias import SuperBiasStep
from jwst.refpix import RefPixStep
from jwst.dark_current import DarkCurrentStep
from jwst.jump import JumpStep
from jwst.ramp_fitting import RampFitStep
from jwst.assign_wcs import AssignWcsStep
from jwst.photom import PhotomStep
from jwst.flatfield.flat_field_step import FlatFieldStep
from jwst.extract_1d.extract_1d_step import Extract1dStep
from jwst.extract_2d.extract_2d_step import Extract2dStep
try:
    import gen_cfgs as cfg
except ImportError:
    from . import gen_cfgs as cfg

warnings.simplefilter('ignore', category=AstropyWarning)

def _class(step):
    """
    Return the pipeline class for each step
    """
    step_dict = {'dq_init':      DQInitStep,     
                 'saturation':   SaturationStep, 
                 'ipc':          IPCStep,        
                 'superbias':    SuperBiasStep,  
                 'refpix':       RefPixStep,     
                 'linearity':    LinearityStep,  
                 'dark_current': DarkCurrentStep,
                 'jump':         JumpStep,       
                 'ramp_fit':     RampFitStep,    
                 'assign_wcs':   AssignWcsStep,  
                 'flat_field':   FlatFieldStep,  
                 'photom':       PhotomStep,     
                 'extract_1d':   Extract1dStep}

    return step_dict[step]
    
def _plot(file, frame=0, log_scale=True, inspect=False):
    """
    Plot a frame of the data in the given file
    
    Parameters
    ----------
    file: str, array-like
        The absolute file path
    frame: int
        The index of the frame to plot
    log_scale: bool
        Use logarithmic scale
    inspect: tuple, list
        The (x,y) coordinates of the point to inspect
    
    """
    # Get one 2D image
    if isinstance(file, str):
        data = fits.getdata(file, 0)
    else:
        data = file
    dims = data.shape
    if len(dims)==4:
        data = data[0][frame]
    elif len(dims)==3:
        data = data[frame]
    
    # Plot it
    plt.figure(figsize=(15,5))
    try:
        plt.imshow(data, origin='lower', interpolation='none',  
           norm=LogNorm() if log_scale else None, vmin=1, vmax=65535)
        X, Y = plt.xlim(), plt.ylim()
        
        if inspect:
            plt.plot([inspect[0]],[inspect[1]], marker='o', color='w')
            print(data[inspect[1]][inspect[0]])
            plt.xlim(X), plt.ylim(Y)
    except:
        print('Could not plot data of shape',dims)
        plt.close()

def calibrate(file, template, arr, levels=['1A', '2A'], skip=[], 
              destination='', plot=True, flipy=False, **kwargs):
    """
    Run the given file through the indicated JWST calibration pipeline 
    levels. By default, all steps and all levels are included.
        
    Parameters
    ----------
    file: str
        The absolute path of the file to calibrate
    template: str
        The path to the template file
    arr: str
        The subarray name, 'SUBSTRIP96', 'SUBSTRIP256', or 'FULL'
    levels: list
        The calibration pipeline levels to complete
    skip: list
        The names of the steps to skip, e.g.
        ['refpix','extract_1d']
    destination: str
        The target directory for the output file
    plot: bool
        Plot a frame of the results
    flipy: bool
        Flip the y-axis data
    
    Returns
    -------
    str
        The output file path
    """
    # List the steps in level 2
    L2 = [['dq_init', 'saturation', 'ipc', 'superbias', 'refpix', 
           'linearity','dark_current', 'jump', 'ramp_fit'],
          ['assign_wcs', 'flat_field', 'photom', 'extract_1d']]
    
    # Convert to pipeline readable file
    if '1A' in levels:
        file = to_DMS(fits.getdata(file), template, arr, destination, nint=3, plot=plot, flipy=flipy)
    
    # Loop through steps in pipeline levels 2A, then 2B
    for n,level in enumerate(['2A','2B']):
        if level in levels:
            for step in L2[n]:
                if step not in skip:
                    try:
                        # Get the step-specific kwargs
                        step_args = kwargs.get(step, {})

                        # Run the step
                        file = run_step(file, step, destination=destination, plot=plot, **step_args)
                    
                    except TypeError:
                        print('Could not complete {} step.'.format(step))
                
    return file
        
def CV3toDMS(file, arr, destination='', plot=False):
    """
    Convert a CV3 file to DMS coordinates with 
    pipeline-appropriate keywords
    
    Parameters
    ----------
    file: str
        The absolute path of the CV3 file to convert
    arr: str
        The FITS header SUBARRAY value
    destination: str
        The target directory for the output file
    plot: bool
        Plot a frame of the results
    
    Returns
    -------
    str
        The output file path
    """
    # Open the file and get the data
    filedark = fits.open(file)
    
    # Get the header keywords
    tlscp = filedark[0].header['TELESCOP']
    inst = filedark[0].header['INSTRUME']
    det = filedark[0].header['DETECTOR']
    filt = filedark[0].header['FWCCRFIL']
    pup = filedark[0].header['PWCCRPUP']
    nfrm = filedark[0].header['NFRAME']
    nint = filedark[0].header['NINT']
    ngrp = filedark[0].header['NGROUP']
    grpgp = filedark[0].header['GROUPGAP']
    tfrm = filedark[0].header['TFRAME']
    tgrp = filedark[0].header['TGROUP']
    inttme = filedark[0].header['INTTIME']
    exptme = filedark[0].header['EXPTIME']
    rdout = filedark[0].header['READOUT']
    #rwcrnr = filebias[0].header['ROWCORNR']
    #clcrnr = filebias[0].header['COLCORNR']
    #nrws = filebias[0].header['NROWS']
    #ncls = filebias[0].header['NCOLS']    
    
    # Get the image shape    
    orig_data = filedark[0].data
    dims = orig_data.shape
    ax1, ax2 = dims[-2:]
    
    # Get the number of groups and integrations
    if len(dims)==4:
        nints, ngrps = dims[:2]
    if len(dims)==3:
        nints, ngrps = nint, dims[0]/nint
    if len(dims)==2:
        nints, ngrps = 1, 1
        
    # Correct dimensions
    dims = list(map(int,[nints, ngrps, ax1, ax2]))
    orig_data = orig_data.reshape(dims)

    # Convert to DMS coordinates
    orig_data = orig_data.swapaxes(-1,-2)
    sci_array = orig_data[:,:,::-1,::-1]
    
    # Create a new JWST data model using the data array
    new_model = datamodels.RampModel(data=sci_array.copy())
    
    # Get the subarray location on the detector
    pix = subarray(arr)
    
    # Set meta data values for header keywords
    new_model.meta.telescope = tlscp
    new_model.meta.instrument.name = inst
    new_model.meta.instrument.detector = 'NIS'
    new_model.meta.instrument.filter = filt
    new_model.meta.instrument.pupil = 'CLEARP'
    new_model.meta.exposure.type = 'NIS_SOSS'
    new_model.meta.exposure.nints = nint
    new_model.meta.exposure.ngroups = ngrp
    new_model.meta.exposure.nframes = nfrm
    new_model.meta.exposure.readpatt = 'NISRAPID'
    new_model.meta.exposure.groupgap = grpgp
    new_model.meta.subarray.name = arr
    new_model.meta.subarray.xsize = sci_array.shape[3]
    new_model.meta.subarray.ysize = sci_array.shape[2]
    new_model.meta.subarray.xstart = pix.get('xloc', 1)
    new_model.meta.subarray.ystart = pix.get('yloc', 1)
    new_model.meta.subarray.fastaxis = -2
    new_model.meta.subarray.slowaxis = -1
    new_model.meta.observation.date=filedark[0].header['DATE-OBS']
    new_model.meta.observation.time=filedark[0].header['TIME-OBS']
    
    # Save the new data model to a FITS file
    path = destination or os.path.dirname(file)+'/'
    name = path+os.path.basename(file).replace(".fits", "_dms.fits")
    new_model.save(name)
    
    # Update PRIMARY extension header
    fits.setval(name, 'TFRAME', value=float(tfrm), ext=0, 
                comment='Time in seconds between frames (sec)')
    fits.setval(name, 'TGROUP', value=float(tgrp), ext=0, 
                comment='Delta time between groups (sec)')
    fits.setval(name, 'INTTIME', value=float(inttme), ext=0, 
                comment='Total integration time for one MULTIACCUM (sec)')
    fits.setval(name, 'EXPTIME', value=float(exptme), ext=0, 
                comment='Exposure duration calculated (sec)')
    fits.setval(name, 'READOUT', value=rdout, ext=0, 
                comment='Readout pattern name')
    fits.setval(name, 'WCSAXES', value=2, ext=0, 
                comment='The number of World Coordinate System axes')
    fits.setval(name, 'CTYPE1', value="RA---TAN", ext=0, 
                comment='First axis coordinate type')
    fits.setval(name, 'CTYPE2', value="DEC--TAN", ext=0, 
                comment='Second axis coordinate type')
    fits.setval(name, 'TARG_RA', value=6.12816, ext=0, 
                comment='Target RA at mid time of exposure')    
    fits.setval(name, 'TARG_DEC', value=-66.597983, ext=0, 
                comment='Target Dec at mid time of exposure')
    
    # Update SCI extension header
    fits.setval(name, 'WCSAXES', value=2, ext=1, 
                comment='The number of World Coordinate System axes')
    fits.setval(name, 'CTYPE1', value="RA---TAN", ext=1, 
                comment='First axis coordinate type')
    fits.setval(name, 'CTYPE2', value="DEC--TAN", ext=1, 
                comment='Second axis coordinate type')
    filedark.close()
    
    if plot:
        _plot(name)
    
    return name

def data_triage(data, filename, err=[], jdmid=[], nints='', grptme=1, 
                subarr='', ndims=4, header=[]):
    '''
    Writes data cube or FITS file to a FITS file with
    all necessary keywords for pipeline ingestion
    
    Parameters
    ----------
    data: array-like
        The input data of at least 2 dimensions
    filename: str
        The filename of the new FITS file
    err: array-like (optional)
        The error array of the same shape as data
    jdmid: array-like (optional)
        The array of Julian dates at integration midpoints
    nints: int (optional)
        The number of integrations
    grptme: float
        The time for each group readout in seconds
    subarr: str
        The subarray shortname
    ndims: int
        The desired number of dimensions for the data,
        3 (nints*ngroups,y,x) or 4 (nints,ngroups,y,x)
    header: list (optional)
        The (key,value,comment) info to put in the header
    
    History
    -------
    Written by Joe Filippazzo      October 2016
    '''
    # If it is a FITS file, get the header and data
    if isinstance(data, str):
        
        # Get the flux data
        HDU = fits.open(data)
        if isinstance(HDU[0].data, np.ndarray):
            data = HDU[0].data
        else:
            data = HDU['SCI'].data
        
        # Try to get the error data
        try:
            err = HDU['ERR'].data
        except:
            pass
        
        # Try to get the JD data
        try:
            jdmid = HDU['JDMID'].data
        except:
            pass
        
        # Get the header
        old_hdr = list(HDU[0].header.cards)
        
        HDU.close()
    
    else:
        old_hdr = []
    
    # Print warning if err and data are different shapes
    if isinstance(err, np.ndarray):
        if err.shape!=data.shape:
            print('Warning: *err* array not the same shape as *data* array.')          
    
    # Get the image shape
    dims = data.shape
    ax1, ax2 = dims[-2:]
    
    # Get subarray size
    subarr = subarr or '96' if ax1==96 else '256'
    
    # Get the number of groups and integrations
    if len(dims)==4:
        nints, ngrps = dims[:2]
    if len(dims)==3:
        if nints:
            ngrps = dims[0]/nints
        else:
            nints, ngrps = 1, dims[0]
    if len(dims)==2:
        nints, ngrps = 1, 1
        
    # Correct dimensions
    if ndims==3:
        dims = (int(nints*ngrps), ax1, ax2)
    elif ndims==4:
        dims = (nints, ngrps, ax1, ax2)
    
    dims = [int(d) for d in dims]
    data = data.reshape(dims)
    try:
        err = err.reshape(dims)
    except:
        pass

    # Convert to SSB coordinates if necessary
    if ax1>ax2:
        data = data.swapaxes(-1,-2)
        try:
            err = err.swapaxes(-1,-2)
        except:
            pass
    if ndims==3:
        data = data[:,::-1,::-1]
    elif ndims==4:
        data = data[:,:,::-1,::-1]
    
    # Calculate times
    inttme = int(grptme*ngrps)
    exptme = int(nints*inttme)
    obstme = int(exptme+(nints*grptme))    
        
    # Create the primary HDU
    prihdu = fits.PrimaryHDU()
    prihdu.name = 'PRIMARY'
    hdulist = fits.HDUList([prihdu])
    
    # Write the data to the HDU
    hdulist.append(fits.ImageHDU(data=data,  name='SCI'))
    hdulist.append(fits.ImageHDU(data=err,   name='ERR'))
    hdulist.append(fits.ImageHDU(data=jdmid, name='JDMID'))

    # Default header cards
    default_hdr = [('FILENAME', os.path.basename(filename), 'Name of file'), \
      ('TELESCOP', '', 'Telescope used to acquire data'), \
      ('INSTRUME', '', 'Instrument used to acquire data'), \
      ('DETECTOR', '', 'Detector used to acquire data'), \
      ('FILTER', '', 'Name of filter element used'), \
      ('PUPIL', '', 'Name of pupil element used'), \
      ('EXP_TYPE', '', 'Type of data in the exposure'), \
      ('NINTS', int(nints), 'Number of integrations'), \
      ('NGROUPS', int(ngrps), 'Number of groups'), \
      ('NFRAMES', 1, 'Number of frames per group'), \
      ('GROUPGAP', int(0), 'Number of frames dropped between groups'), \
      ('TFRAME', grptme, 'Time in seconds between frames'), \
      ('TGROUP', grptme, 'Delta time between frames'), \
      ('INTTIME', inttme, 'Total integration time for one MULTIACCUM'), \
      ('EXPTIME', exptme, 'Exposure duration (seconds) calculated'), \
      ('OBSTIME', obstme, 'Observatory dur. (sec., inc. reset time) cal.'), \
      ('SUBARRAY', subarrays(subarr)['SUBARRAY'], 'Subarray of image'), \
      ('SUBSTRT1', int(1), 'Starting pixel in axis 1 direction'), \
      ('SUBSTRT2', subarrays(subarr)['SUBSTRT2'], 'Starting pixel in axis 2 direction'), \
      ('SUBSIZE1', int(ax1), 'Number of pixels in axis 1 direction'), \
      ('SUBSIZE2', int(ax2), 'Number of pixels in axis 2 direction'), \
      ('FASTAXIS', int(-2), 'Fast readout axis direction'), \
      ('SLOWAXIS', int(-1), 'Slow readout axis direction'), \
      ('READPATT', '', 'Readout pattern name')]
    
    # Write the header to the PRIMARY HDU
    for k,v,c in default_hdr+header+old_hdr:
        hdulist['PRIMARY'].header.set(k, value=v, comment=c)
    
    # Write the file
    hdulist.writeto(filename, overwrite=True)
    hdulist.close()

    print('File saved as', filename)

def to_DMS(data, template, arr, destination='', nint=1, plot=True, **kwargs):
    """
    Create a DMS formatted fits file using a pipeline
    compatible template and some data
    
    Parameters
    ----------
    data: str, np.ndarray
        The file to fetch data from or a data cube 
    template: str
        The path to the template file
    arr: str
        The subarray name, i.e. ['FULL', 'SUBSTRIP256', 'SUBSTRIP96']
    destination: str
        The output filepath
    nint: int (optional)
        The number of integrations if the data is in 3D
        instead of 4D
    plot: bool
        Plot a frame for inspection
        
    Returns
    -------
    str
        The new filepath
    """
    # Get the new file location
    if isinstance(data, str):    
        path = destination or os.path.dirname(data)+'/'
        name = path+os.path.basename(data).replace(".fits", "_dms.fits")
    else:
        name = destination+'DMS_data.fits'
    
    # Copy the correctly formatted FITS file to the directory
    copyfile(template, name)
    
    # Open the file
    HDU = fits.open(name)
    
    # Get the subarray metadata
    meta = subarray(arr)
    
    # Get the data
    if isinstance(data, str):
        # See what extensions are in the input file
        hdu = fits.open(data)
        
        # Add data for each extension
        for n,ext in enumerate([i[1] for i in HDU.info(False)]):
            try:
                HDU[ext].data = hdu[ext].data
                HDU[ext].header = hdu[ext].header
            except:
                pass
        
        data = hdu[0].data
        hdu.close()
    else:
        pass
    
    # Get data dimensions
    dims = data.shape
    ax1, ax2 = dims[-2:]
    
    # Get the number of groups and integrations
    if len(dims)==4:
        nints, ngrps = dims[:2]
    if len(dims)==3:
        nints, ngrps = nint, int(dims[0]/nint)
    if len(dims)==2:
        nints, ngrps = 1, 1
    
    # Calculate times
    grptme = meta.get('tgrp')
    frmtme = meta.get('tfrm')
    inttme = int(grptme*ngrps)
    exptme = int(nints*inttme)
    obstme = int(exptme+(nints*grptme)) 
    
    # Correct dimensions
    dims = list(map(int,[nints, ngrps, ax1, ax2]))
    d = data.reshape(dims)
    HDU['SCI'].data = d.swapaxes(-1,-2)[:,:,::-1,::-1]
    for ext in ['GROUPDQ','ERR']:
        try:
            d = HDU[ext].data[:,:,::-1,::-1]
            HDU[ext].data = d[:nints,:ngrps,:ax2,:ax1]
        except IOError:
            pass
    try:
        HDU['PIXELDQ'].data = HDU['PIXELDQ'].data[:ax2,:ax1]
    except:
        pass

    # Update some other keywords
    HDU[0].header['SUBARRAY'] = arr
    HDU[0].header['SUBSTRT1'] = meta.get('xloc', 1)
    HDU[0].header['SUBSTRT2'] = meta.get('yloc', 1)
    HDU[0].header['SUBSIZE1'] = meta.get('x', 2048)
    HDU[0].header['SUBSIZE2'] = meta.get('y', 2048)
    HDU[0].header['NINTS'] = nints
    HDU[0].header['NGROUPS'] = ngrps
    HDU[0].header['EFFINTTM'] = inttme
    HDU[0].header['EFFEXPTM'] = exptme
    HDU[0].header['TFRAME'] = frmtme
    HDU[0].header['TGROUP'] = grptme
    HDU[0].header['TARG_RA'] = 6.12816
    HDU[0].header['TARG_DEC'] = -66.597983
    HDU[0].header['FILTER'] = 'CLEAR'
    HDU[0].header['PUPIL'] = 'GR700XD'
    HDU[0].header['ORDER'] = 1

    # Update keywords from kwargs
    for k,v in kwargs.items():
        HDU[0].header[k] = v
    
    # Write the file
    HDU.writeto(name, overwrite=True)
    HDU.close()
    
    if plot:
        _plot(name)
        
    return name
    
def info(step):
    """
    Print some info about the step
    
    Parameters
    ----------
    step: str
        The pipeline step to inspect
    
    """
    # Get the step Class attributes and then delete the object
    step = _class(step)()
    i = vars(step)
    
    # Print the docstring description
    print(step.__doc__)
    
    # Clean up memory
    del step
    
    # Print the values in a little table
    l1 = max([len(k) for k in i.keys()])
    l2 = max([len(repr(v)) for k,v in i.items()])
    print("{:<{}} {}".format('Attribute', l1, 'Default Value'))
    print("{:<{}} {}".format('-'*l1, l1, '-'*l2))
    for k,v in i.items():
        if not k.startswith('_'):
            print("{:<{}} {}".format(k, l1, repr(v)))

def run_step(file, step, destination='', plot=True, **kwargs):
    """
    Perform the given *step* of the JWST pipeline and
    create a new file of the calibrated data
    
    Parameters
    ----------
    file: str
        The absolute path of the file to calibrate
    step: str
        The pipeline step to execute
    destination: str
        The target directory for the output file
    plot: bool
        Plot a frame of the results
    
    Returns
    -------
    str
        The output file path
    """
    # Create new filename
    path = destination or os.path.dirname(file)+'/'
    name = path+os.path.basename(file).replace(".fits", "_{}.fits".format(step))
    
    # Generate the config file
    cf = getattr(cfg, step+'_cfg')(destination)
    
    try:
        # Run the step
        _class(step).call(file, config_file=cf, output_file=name, **kwargs)
    
        # Plot it
        if plot:
            _plot(name)
            fits.info(name)
    
    except:
        name = ''
    
    # Delete the config file
    os.remove(cf)
    
    return name

#def run_step(data, step, save='', plot=True, **kwargs):
#    """
#    Perform the given *step* of the JWST pipeline and
#    create a new file of the calibrated data
#    
#    Parameters
#    ----------
#    data: DataModel, str
#        The DataModel or absolute path of the file to calibrate
#    step: str
#        The pipeline step to execute
#    save: str (optional)
#        The target directory for the output FITS file
#    plot: bool
#        Plot a frame of the results
#    
#    Returns
#    -------
#    str
#        The output file path
#    """    
#    # Test if filepath or DataModel
#    
#    try:
#        # Run the step
#        STEP = _class(step)(**kwargs)
#        data = STEP.run(data, **kwargs)
#    
#    except IOError:
#        print(step,'step not completed.') 
#    
#    if save:
#        data.save(save)
#    else:
#        return data
#
#def calibrate(data, save='', plot=False,
#              steps=['dq_init', 'saturation', 'ipc', 'superbias', 'refpix', 
#                     'linearity','dark_current', 'jump', 'ramp_fit',
#                     'assign_wcs', 'flat_field', 'photom', 'extract_1d'],
#              **kwargs):
#    """
#    Run the given file through the indicated JWST calibration pipeline 
#    levels. By default, all steps and all levels are included.
#        
#    Parameters
#    ----------
#    data: DataModel, str
#        The DataModel objects or absolute path of the file to calibrate
#    steps: list
#        The calibration pipeline steps to complete
#    save: str (optional)
#        The target directory for the output file
#    plot: bool
#        Plot a frame of the results
#    
#    Returns
#    -------
#    str
#        The output file path
#    """    
#    # Loop through steps in pipeline levels 2A, then 2B
#    for step in steps:
#        
#        try:
#        
#            # Get the step-specific kwargs
#            step_args = kwargs.get(step, {})
#
#            # Run the step
#            data = run_step(data, step, **step_args)
#        
#        except IOError:
#            print('Could not complete {} step.'.format(step))
#            
#        # Save when done
#        if save:
#            data.save(save)
#            
#        else:
#            return file 
    
def subarray(arr=''):
    """
    Get the pixel information for each NIRISS subarray.     
    
    The returned dictionary defines the extent ('x' and 'y'),
    the starting pixel ('xloc' and 'yloc'), and the number 
    of reference pixels at each subarray edge ('x1', 'x2',
    'y1', 'y2) as defined by SSB/DMS coordinates shown below:
        ___________________________________
       |               y2                  |
       |                                   |
       |                                   |
       | x1                             x2 |
       |                                   |
       |               y1                  |
       |___________________________________|
    (1,1)
    
    Parameters
    ----------
    arr: str
        The FITS header SUBARRAY value
    
    Returns
    -------
    dict
        The dictionary of the specified subarray
        or a nested dictionary of all subarrays
    
    """
    pix = {'FULL'        : {'xloc':1,    'x':2048, 'x1':4, 'x2':4,
                            'yloc':1,    'y':2048, 'y1':4, 'y2':4,
                            'tfrm':10.73676, 'tgrp':10.73676},
           'SUBSTRIP96'  : {'xloc':1,    'x':2048, 'x1':4, 'x2':4,
                            'yloc':1803, 'y':96,   'y1':0, 'y2':0,
                            'tfrm':2.3, 'tgrp':2.3},
           'SUBSTRIP256' : {'xloc':1,    'x':2048, 'x1':4, 'x2':4,
                            'yloc':1793, 'y':256,  'y1':0, 'y2':4,
                            'tfrm':5.4, 'tgrp':5.4},
           'SUB80'       : {'xloc':None, 'x':80,   'x1':0, 'x2':0,
                            'yloc':None, 'y':80,   'y1':4, 'y2':0},
           'SUB64'       : {'xloc':None, 'x':64,   'x1':0, 'x2':4,
                            'yloc':None, 'y':64,   'y1':0, 'y2':4},
           'SUB128'      : {'xloc':None, 'x':128,  'x1':0, 'x2':4,
                            'yloc':None, 'y':128,  'y1':0, 'y2':4},
           'SUB256'      : {'xloc':None, 'x':256,  'x1':0, 'x2':4,
                            'yloc':None, 'y':256,  'y1':0, 'y2':4},
           'SUBAMPCAL'   : {'xloc':None, 'x':512,  'x1':4, 'x2':0,
                            'yloc':None, 'y':1792, 'y1':4, 'y2':0},
           'WFSS64R'     : {'xloc':None, 'x':64,   'x1':0, 'x2':4,
                            'yloc':1,    'y':2048, 'y1':4, 'y2':0},
           'WFSS64C'     : {'xloc':1,    'x':2048, 'x1':4, 'x2':0,
                            'yloc':None, 'y':64,   'y1':0, 'y2':4},
           'WFSS128R'    : {'xloc':None, 'x':128,  'x1':0, 'x2':4,
                            'yloc':1,    'y':2048, 'y1':4, 'y2':0},
           'WFSS128C'    : {'xloc':1,    'x':2048, 'x1':4, 'x2':0,
                            'yloc':None, 'y':128,  'y1':0, 'y2':4},
           'SUBTASOSS'   : {'xloc':None, 'x':64,   'x1':0, 'x2':0,
                            'yloc':None, 'y':64,   'y1':0, 'y2':0},
           'SUBTAAMI'    : {'xloc':None, 'x':64,   'x1':0, 'x2':0,
                            'yloc':None, 'y':64,   'y1':0, 'y2':0}}
    
    try:
        return pix[arr]
    except:
        return pix
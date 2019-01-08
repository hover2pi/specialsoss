#!/usr/bin/env pytho
"""
This module generates all the necessary config files to run the JWST pipeline
"""

def assign_wcs_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "assign_wcs"
class = "jwst.assign_wcs.AssignWcsStep"
"""
    
    # Write the complete text to file
    f = destination+'assign_wcs.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)
        
    return f
        
def ipc_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "ipc" 
class = "jwst.ipc.IPCStep"
"""
    
    # Write the complete text to file
    f = destination+'ipc.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)
        
    return f
        
def flat_field_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "flat_field" 
class = "jwst.flatfield.flat_field_step.FlatFieldStep"

# Optional filename suffix for output flats (only for MOS data).
flat_suffix = None
"""
    
    # Write the complete text to file
    f = destination+'flat_field.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)
        
    return f
    
def jump_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "jump"
class = "jwst.jump.JumpStep"
rejection_threshold = 5.0
do_yintercept = False
yint_threshold = 1.0
"""
    
    # Write the complete text to file
    f = destination+'jump.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)
        
    return f

def linearity_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "linearity"
class = "jwst.linearity.LinearityStep"
"""
    
    # Write the complete text to file
    f = destination+'linearity.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)
        
    return f
           
def photom_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "photom"
class = "jwst.photom.PhotomStep"
"""
    
    # Write the complete text to file
    f = destination+'photom.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)  
        
    return f
        
def ramp_fit_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "RampFit"
class = "jwst.ramp_fitting.RampFitStep"
save_opt = False
opt_name = ""
int_name = "" 
algorithm = "OLS"
"""
    
    # Write the complete text to file
    f = destination+'ramp_fit.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)        
        
    return f
        
def refpix_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "refpix"
class = "jwst.refpix.RefPixStep"
odd_even_columns = True
use_side_ref_pixels = True
side_smoothing_length=11
side_gain=1.0
odd_even_rows = True
"""
    
    # Write the complete text to file
    f = destination+'refpix.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)        
        
    return f

def saturation_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "saturation"
class = "jwst.saturation.SaturationStep"
"""
    
    # Write the complete text to file
    f = destination+'saturation.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)     
    
    return f

def superbias_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "superbias" 
class = "jwst.superbias.SuperBiasStep"
"""
    
    # Write the complete text to file
    f = destination+'superbias.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)     
        
    return f
        
def extract_2d_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "extract_2d" 
class = "jwst.extract_2d.Extract2dStep"
"""
    
    # Write the complete text to file
    f = destination+'extract_2d.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)  
    
    return f
                
def extract_1d_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "extract_1d"
class = "jwst.extract_1d.Extract1dStep"

smoothing_length = 0
"""
    
    # Write the complete text to file
    f = destination+'extract_1d.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)         
    
    return f
        
def dq_init_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "dq_init"
class = "jwst.dq_init.DQInitStep"
"""
    
    # Write the complete text to file
    f = destination+'dq_init.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)   
        
    return f
        
def dark_current_cfg(destination='./'):
    # Add the strings to the complete text
    text = """name = "dark_current" 
class = "jwst.dark_current.DarkCurrentStep"
"""
    
    # Write the complete text to file
    f = destination+'dark_current.cfg'
    with open(f,'w') as cfgfile:
        cfgfile.write(text)
    
    return f

def calniriss_2A_cfg(destination='./', subarr='SUBSTRIP256', **kwargs):
    """
    Generate a .cfg file for the level 2A calibration step
    """
    # Dictionary to hold the config file values
    values = {}
    y = 96 if subarr=='SUBSTRIP96' else 256
    
    # Define the default values
    defaults = {'dq_init':{'skip':False, 'override_mask':'niriss_ref_bad_pixel_mask.fits'},
                'saturation':{'skip':False, 'override_saturation':'niriss_ref_saturation.fits'},
                'ipc':{'skip':False, 'override_ipc':'niriss_ref_ipc.fits'},
                'superbias':{'skip':False, 'override_superbias':'niriss_ref_nisrapid_bias_{}_2048.fits'.format(y)},
                'refpix':{'skip':False},
                'linearity':{'skip':False, 'override_linearity':'niriss_ref_linearity.fits'},
                'dark_current':{'skip':False, 'override_dark':'niriss_ref_nisrapid_dark_{}_2048.fits'.format(y)},
                'jump':{'skip':False, 'override_gain':'niriss_ref_gain.fits', 'override_readnoise':'niriss_ref_rdns_dn.fits'},
                'ramp_fit':{'skip':False, 'save_opt':False, 'override_gain':'niriss_ref_gain.fits', 'override_readnoise':'niriss_ref_rdns_dn.fits'}
               }
    
    # Update the defaults if necessary and create strings
    for step,default in defaults.items():
        defaults[step].update({'config_file':step+'.cfg'})
        if step in kwargs:
            defaults[step].update(kwargs[step])
        values[step] = '[[{}]]\n\t\t'.format(step)\
                       +'skip = {}\n\t\t'.format(defaults[step]['skip'])\
                       +'\n\t\t'.join(["{} = {}".format(k,v) for k,v in defaults[step].items() if k!='skip'])

    # Add the strings to the complete text
    text = """name = "SloperPipeline"
class = "jwst.pipeline.SloperPipeline"
save_calibrated_ramp = True

    [steps]
      {dq_init}
      {saturation}
      {ipc}
      {superbias}
      {refpix}
      {linearity}
      {dark_current}
      {jump}
      {ramp_fit}
"""
    
    # Write the complete text to file
    with open(destination+'calniriss_2A.cfg','w') as cfgfile:
        cfgfile.write(text.format(**values))
        
    # Write the dependent config files too
    cfg_paths = [destination+'calniriss_2A.cfg']
    cfg_paths.append(dq_init_cfg(destination))
    cfg_paths.append(saturation_cfg(destination))
    cfg_paths.append(ipc_cfg(destination))
    cfg_paths.append(superbias_cfg(destination))
    cfg_paths.append(refpix_cfg(destination))
    cfg_paths.append(linearity_cfg(destination))
    cfg_paths.append(dark_current_cfg(destination))
    cfg_paths.append(jump_cfg(destination))
    cfg_paths.append(ramp_fit_cfg(destination))
    
    return cfg_paths  

def calniriss_2B_cfg(destination='./', subarr='SUBSTRIP256', **kwargs):
    """
    Generate a .cfg file for the level 2Ab calibration step
    """
    # Dictionary to hold the config file values
    values = {}
    y = 96 if subarr=='SUBSTRIP96' else 256
    
    # Define the default values
    defaults = {'assign_wcs':{'skip':False, 'override_distortion':'niriss_ref_distortion_image.asdf', 'override_specwcs':'niriss_ref_wave_cal.asdf'},
                'extract_2d':{'skip':False},
                'flat_field':{'skip':True}, #'override_ipc':'niriss_ref_ipc.fits'
                'photom':{'skip':True, 'override_photom':'niriss_ref_photom.fits', 'override_area':'niriss_ref_pxl_area.fits'},
                'extract_1d':{'skip':True}
               }
    
    # Update the defaults if necessary and create strings
    for step,default in defaults.items():
        defaults[step].update({'config_file':step+'.cfg'})
        if step in kwargs:
            defaults[step].update(kwargs[step])
        values[step] = '[[{}]]\n\t\t'.format(step)\
                       +'skip = {}\n\t\t'.format(defaults[step]['skip'])\
                       +'\n\t\t'.join(["{} = {}".format(k,v) for k,v in defaults[step].items() if k!='skip'])

    # Add the strings to the complete text
    text = """name = "Spec2Pipeline"
#class = "jwst.pipeline.Spec2Pipeline"
class = "jwst.pipeline.Spec2Pipeline"
save_bsub = False

    [steps]
      {assign_wcs}
      {extract_2d}
      {flat_field}
      {photom}
      {extract_1d}
"""
    
    # Write the complete text to file
    with open(destination+'calniriss_2B.cfg','w') as cfgfile:
        cfgfile.write(text.format(**values))
        
    # Write the dependent config files too
    cfg_paths = [destination+'calniriss_2B.cfg']
    cfg_paths.append(assign_wcs_cfg(destination))
    cfg_paths.append(extract_2d_cfg(destination))
    cfg_paths.append(flat_field_cfg(destination))
    cfg_paths.append(photom_cfg(destination))
    cfg_paths.append(extract_1d_cfg(destination))
    
    return cfg_paths
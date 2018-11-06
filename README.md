# `specialsoss`: SPECtral Image AnaLysis for SOSS

[![Build Status](https://travis-ci.org/hover2pi/specialsoss.svg?branch=master)](https://travis-ci.org/hover2pi/specialsoss)

Authors: Joe Filippazzo, Kevin Stevenson

This pure Python package performs optimal spectral extraction routines for the Single Object Slitless Spectroscopy (SOSS) mode of the Near-Infrared Imager and Slitless Spectrograph (NIRISS) instrument onboard the James Webb Space Telescope (JWST).

## Dependencies

The following packages are needed to run `specialsoss`:
- numpy
- astropy
- scipy

## Extracting Spectra from SOSS Observations

The headers in JWST data products provide almost all the information needed to perform the spectral extraction, making the path to your data the only required input. To extract time series 1D spectra, simply do

```
# Imports
import numpy as np
from specialsoss import SossObs
from pkg_resources import resource_filename

# Run the extraction
data = resource_filename('specialsoss', 'files/soss_example.fits')
obs = SossObs(data)
```

That's it! Now we can take a look at the extracted time-series spectra:

```
obs.plot()
```

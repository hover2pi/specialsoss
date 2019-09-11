# -*- coding: utf-8 -*-

"""A module for the 1D spectral extraction binning method"""

import copy

from astropy.io import fits
import astropy.units as q
from bokeh.plotting import figure, show
import numpy as np
from pkg_resources import resource_filename

from hotsoss import locate_trace as lt
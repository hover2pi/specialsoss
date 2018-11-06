#!/usr/bin/env python
from setuptools import setup

setup(name='specialsoss',
      version=0.1,
      description='Spectral Image Analysis for SOSS',
      install_requires=['astropy', 'scipy', 'matplotlib', 'numpy'],
      author='Joe Filippazzo and Kevin Stevenson',
      author_email='jfilippazzo@stsci.edu',
      license='MIT',
      url='https://github.com/hover2pi/specialsoss/',
      long_description='SPECial SOSS performs optimal 1D spectral extraction of time series data for the Single Object Slitless Spectroscopy mode of the NIRISS instrument onboard JWST',
      zip_safe=False,
      use_2to3=False
)

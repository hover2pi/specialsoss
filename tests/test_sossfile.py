#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sossfile` module."""

import unittest
from pkg_resources import resource_filename

from specialsoss import sossfile


class TestSossFile(unittest.TestCase):
    """Test SossFile object"""
    def setUp(self):
        """Test instance setup"""
        # Get files for testing
        self.uncal = resource_filename('specialsoss', 'files/SUBSTRIP256_CLEAR_uncal.fits')
        self.rateints = resource_filename('specialsoss', 'files/SUBSTRIP256_CLEAR_rateints.fits')

    def test_init(self):
        """Test that files can be initialized"""
        # uncal check
        obs = sossfile.SossFile(self.uncal)
        obs.file = None

        # Test properties
        print(obs)

    def test_calibrate(self):
        """Test calibrate method"""
        obs_uncal = sossfile.SossFile(self.uncal)

        # See if jwst is installed
        try:
            import jwst

            # Calibrate uncal
            new_files1 = obs_uncal.calibrate()

            # Calibrate rateints
            obs_rateints = sossfile.SossFile(new_files['rateints'])
            new_files2 = obs_rateints.calibrate()

            # Try to calibrate '_ramp.fits' file but print message instead
            obs_ramp = sossfile.SossFile(new_files1['ramp'])
            obs_ramp.calibrate()

        except ModuleNotFoundError:
            self.assertRaises(ModuleNotFoundError, obs_uncal.calibrate)

    def test_plot(self):
        """Test plot method"""
        obs = sossfile.SossFile(self.uncal)
        fig = obs.plot()

        # Check the figure type
        self.assertEqual(str(type(fig)), "<class 'bokeh.models.layouts.Column'>")

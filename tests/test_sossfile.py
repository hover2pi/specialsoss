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

        # rateints check
        obs = sossfile.SossFile(self.rateints)

        # Test properties
        print(obs)

    def test_plot(self):
        """Test plot method"""
        obs = sossfile.SossFile(self.uncal)
        fig = obs.plot()

        # Check the figure type
        self.assertEqual(str(type(fig)), "<class 'bokeh.models.layouts.Column'>")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `specialsoss` package."""

import unittest
from pkg_resources import resource_filename

import numpy as np
import astropy.units as q
import bokeh

from specialsoss import specialsoss


class TestSossObs(unittest.TestCase):
    """Test SossObs object"""
    def setUp(self):
        """Test instance setup"""
        # Make Spectrum class for testing
        self.file = resource_filename('specialsoss', 'files/SOSS_A0.fits')

    def test_init(self):
        """Test that a purely photometric SED can be creted"""
        obs = specialsoss.SossObs(filepath=self.file)
        assert obs.name == 'My SOSS Observations'

    def test_plot(self):
        """Check the plotting works"""
        obs = specialsoss.SossObs(filepath=self.file)
        fig = obs.plot_frame(draw=False)
        assert str(type(fig)) == "<class 'bokeh.plotting.figure.Figure'>"

    def test_wavecal(self):
        """Test loading wavecal file"""
        obs = specialsoss.SossObs(filepath=self.file)
        assert obs.wavecal.shape == (3, 256, 2048)


class TestTestObs(unittest.TestCase):
    """Test TestObs object"""
    def setUp(self):
        """Test instance setup"""
        pass

    def test_init(self):
        """Test that the test object loads properly"""
        obs = specialsoss.TestObs()
        assert obs.name == 'Test Observation'

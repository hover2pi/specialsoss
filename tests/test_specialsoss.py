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
        self.file = resource_filename('specialsoss', 'files/SOSS256_sim.fits')

    def test_init(self):
        """Test that a purely photometric SED can be creted"""
        # Check name
        obs = specialsoss.SossObs(self.file)
        self.assertEqual(obs.name, 'My SOSS Observations')

        # Check data ingest
        self.assertEqual(obs.datacube.shape, (5, 5, 256, 2048))
        self.assertEqual(obs.subarray, 'SUBSTRIP256')
        self.assertEqual(obs.filter, 'CLEAR')
        self.assertEqual(obs.nints, 5)
        self.assertEqual(obs.ngrps, 5)
        self.assertEqual(obs.nrows, 256)
        self.assertEqual(obs.ncols, 2048)

    def test_info(self):
        """Test the info property"""
        obs = specialsoss.SossObs(self.file)
        obs.info

    def test_plot(self):
        """Check the plotting works"""
        obs = specialsoss.SossObs(self.file)
        fig = obs.plot_frame(draw=False)
        self.assertEqual(str(type(fig)), "<class 'bokeh.plotting.figure.Figure'>")

    def test_wavecal(self):
        """Test loading wavecal file"""
        obs = specialsoss.SossObs(self.file)
        self.assertEqual(obs.wavecal.shape, (3, obs.nrows, obs.ncols))


class TestSimObs(unittest.TestCase):
    """Test TestObs object"""
    def setUp(self):
        """Test instance setup"""
        pass

    def test_init(self):
        """Test that the test object loads properly"""
        obs = specialsoss.SimObs()
        self.assertEqual(obs.name, 'Simulated Observation')

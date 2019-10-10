#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `specialsoss` package."""

import unittest
from pkg_resources import resource_filename

import numpy as np
import astropy.units as q
import bokeh

from specialsoss import specialsoss


class TestSossExposure(unittest.TestCase):
    """Test SossExposure object"""
    def setUp(self):
        """Test instance setup"""
        # Make Spectrum class for testing
        self.file = resource_filename('specialsoss', 'files/SUBSTRIP256_CLEAR_ramp.fits')

    def test_init(self):
        """Test that a purely photometric SED can be creted"""
        # Check name
        obs = specialsoss.SossExposure(self.file, calibrate=False)
        self.assertEqual(obs.name, 'My SOSS Observations')

        # Check data ingest
        self.assertEqual(obs.data.shape, (2, 2, 256, 2048))
        self.assertEqual(obs.subarray, 'SUBSTRIP256')
        self.assertEqual(obs.filter, 'CLEAR')
        self.assertEqual(obs.nints, 2)
        self.assertEqual(obs.ngrps, 2)
        self.assertEqual(obs.nrows, 256)
        self.assertEqual(obs.ncols, 2048)

    def test_info(self):
        """Test the info property"""
        obs = specialsoss.SossExposure(self.file, calibrate=False)
        obs.info

    def test_get_frame(self):
        """Test the _get_frame hidden method"""
        # Make CLEAR obs
        clear = specialsoss.SossExposure(self.file, calibrate=False)

        # with idx
        dat = clear._get_frame(idx=0)
        self.assertEqual(np.sum(dat), np.sum(clear.data[0][0]))

        # withoug idx, get median
        dat = clear._get_frame()
        self.assertEqual(np.sum(dat), np.sum(clear.median))

    def test_plot(self):
        """Check the plotting works"""
        obs = specialsoss.SossExposure(self.file, calibrate=False)
        fig = obs.plot_frame(draw=False)
        self.assertEqual(str(type(fig)), "<class 'bokeh.plotting.figure.Figure'>")

    def test_wavecal(self):
        """Test loading wavecal file"""
        obs = specialsoss.SossExposure(self.file, calibrate=False)
        self.assertEqual(obs.wavecal.shape, (3, obs.nrows, obs.ncols))

    def test_decontaminate(self):
        """Test the decontaminate method works"""
        # Make CLEAR obs
        clear = specialsoss.SossExposure(self.file, calibrate=False)

        # Make F277W obs
        f277w = specialsoss.SossExposure(self.file, calibrate=False)
        f277w.filter = 'F277W'

        # Fail if obs2 is not SossExposure
        self.assertRaises(TypeError, clear.decontaminate, 'FOO')

        # Fail if obs2.filter is not F277W
        self.assertRaises(ValueError, clear.decontaminate, clear)

        # Fail if obs1.fiter is not CLEAR
        self.assertRaises(ValueError, f277w.decontaminate, f277w)

        # Fail if obs1 is not extracted
        self.assertRaises(ValueError, clear.decontaminate, f277w)
        clear.extract('sum')

        # Fail if obs2 is not extracted
        self.assertRaises(ValueError, clear.decontaminate, f277w)
        f277w.extract('sum')

        # Run decontaminate
        clear.decontaminate(f277w)

    def test_extract(self):
        """Test the extract method"""
        # Make CLEAR obs
        clear = specialsoss.SossExposure(self.file, calibrate=False)

        # Fail if bad etraction method
        self.assertRaises(ValueError, clear.extract, 'FOO')

        # Check extracted is empty
        self.assertEqual(clear.extracted, {})

        # Run and check extracted is populated
        clear.extract('sum')
        self.assertNotEqual(clear.extracted, {})

    def test_plots(self):
        """Test the plots work"""
        # Make CLEAR obs
        clear = specialsoss.SossExposure(self.file, calibrate=False)

        # Test plot_frame
        fig = clear.plot_frame(draw=False)

        # Test plot_ramp
        fig = clear.plot_ramp(draw=False)

        # Test plot_slice
        fig = clear.plot_slice(col=500, draw=False)

        # Test plot_spectra
        fig = clear.plot_spectra(dtype='flux', draw=False)
        fig = clear.plot_spectra(dtype='counts', draw=False)


class TestSimExposure(unittest.TestCase):
    """Test SimExposure object"""
    def setUp(self):
        """Test instance setup"""
        pass

    def test_init(self):
        """Test that the test object loads properly"""
        obs = specialsoss.SimExposure()
        self.assertEqual(obs.name, 'Simulated Observation')

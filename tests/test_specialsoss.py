#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `specialsoss` module."""

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
        # Get files for testing
        self.uncal = resource_filename('specialsoss', 'files/SUBSTRIP256_CLEAR_uncal.fits')
        self.rateints = resource_filename('specialsoss', 'files/SUBSTRIP256_CLEAR_rateints.fits')

    def test_init(self):
        """Test that a rateints file can be initialized"""
        # Check name
        obs = specialsoss.SossExposure(self.uncal)
        self.assertEqual(obs.name, 'My SOSS Observations')

        # Check data ingest
        self.assertEqual(obs.uncal.data.shape, (2, 2, 256, 2048))
        self.assertEqual(obs.subarray, 'SUBSTRIP256')
        self.assertEqual(obs.filter, 'CLEAR')
        self.assertEqual(obs.nints, 2)
        self.assertEqual(obs.ngrps, 2)
        self.assertEqual(obs.nrows, 256)
        self.assertEqual(obs.ncols, 2048)

    def test_info(self):
        """Test the info property"""
        obs = specialsoss.SossExposure(self.uncal)
        obs.info

    def test_wavecal(self):
        """Test loading wavecal file"""
        obs = specialsoss.SossExposure(self.uncal)
        self.assertEqual(obs.wavecal.shape, (3, 2048, 2048))

    def test_decontaminate(self):
        """Test the decontaminate method works"""
        # Make CLEAR obs
        clear = specialsoss.SossExposure(self.rateints)

        # Make F277W obs
        f277w = specialsoss.SossExposure(self.rateints)
        f277w.filter = 'F277W'

        # Fail if obs2 is not SossExposure
        self.assertRaises(TypeError, clear.decontaminate, 'FOO')

        # Fail if obs2.filter is not F277W
        self.assertRaises(ValueError, clear.decontaminate, clear)

        # Fail if obs1.fiter is not CLEAR
        self.assertRaises(ValueError, f277w.decontaminate, f277w)

        # Fail if obs1 is not extracted
        self.assertRaises(ValueError, clear.decontaminate, f277w)
        clear.extract()

        # Fail if obs2 is not extracted
        self.assertRaises(ValueError, clear.decontaminate, f277w)
        f277w.extract()

        # Run decontaminate
        clear.decontaminate(f277w)

    def test_extract(self):
        """Test the extract method"""
        # Make CLEAR obs
        clear = specialsoss.SossExposure(self.rateints)

        # Fail if bad etraction method
        self.assertRaises(ValueError, clear.extract, 'FOO')

        # Check extracted is empty
        self.assertEqual(clear.results, {})

        # Run and check extracted is populated
        clear.extract()
        self.assertNotEqual(clear.results, {})

    def test_plots(self):
        """Test the plots work"""
        # Make CLEAR obs
        clear = specialsoss.SossExposure(self.uncal)
        clear.extract('sum', 'uncal')

        # Test plot_frames
        fig = clear.plot_frames(draw=False)

        # Test result plot
        fig = clear.plot_results(draw=False)

        # Bad dtype
        self.assertRaises(ValueError, clear.plot_results, 'FOO', draw=True)

        # Test comparison plot
        fig = clear.compare_results(dtype='counts', draw=False)


class TestSimExposure(unittest.TestCase):
    """Test SimExposure object"""
    def setUp(self):
        """Test instance setup"""
        pass

    def test_init(self):
        """Test that the test object loads properly"""
        obs = specialsoss.SimExposure()
        self.assertEqual(obs.name, 'Simulated Observation')

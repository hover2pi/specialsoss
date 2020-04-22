#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `utilities` module."""

import unittest

import numpy as np

from specialsoss import utilities as u


class TestCombineSpectra(unittest.TestCase):
    """Test combine_spectra function"""
    def setUp(self):
        """Test instance setup"""
        # Make spectra for testing
        self.s1 = np.array([np.linspace(0.9, 2.8, 2048), np.random.normal(loc=1000, size=2048), np.random.normal(loc=10, size=2048)])
        self.s2 = np.array([np.linspace(0.6, 1.4, 1632), np.random.normal(loc=1000, size=1632), np.random.normal(loc=10, size=1632)])
        self.s3 = np.array([np.linspace(0.5, 0.88, 1632), np.random.normal(loc=1000, size=1632), np.random.normal(loc=10, size=1632)])

    def testOverlap(self):
        """Test that the spectra can be combined when overlapping"""
        # Run function
        result1 = u.combine_spectra(self.s1, self.s2)

        # Check combined spectra result
        self.assertEqual(result1.ndim, 2)
        self.assertEqual(result1.shape[0], 3)

        # # Check if order matters
        # TODO: This fails right now. Fix it!
        # result2 = u.combine_spectra(self.s2, self.s1)
        # self.assertTrue((result1 == result2).all())

    def testNoOverlap(self):
        """Test that the spectra can be combined when overlapping"""
        # Run function
        result1 = u.combine_spectra(self.s1, self.s3)

        # Check combined spectra result
        self.assertEqual(result1.ndim, 2)
        self.assertEqual(result1.shape[0], 3)

        # Check if order matters
        result2 = u.combine_spectra(self.s3, self.s1)
        self.assertTrue((result1 == result2).all())


class TestNanReferencePixels(unittest.TestCase):
    """Test nan_reference_pixels function"""
    def setUp(self):
        """Test instance setup"""
        pass

    def test2D(self):
        """Test the NaNs are inserted into 2D data"""
        # Make 2D data
        data2048_2D = np.ones((2048, 2048))

        # Run the function
        result = u.nan_reference_pixels(data2048_2D)

        # Check shape and NaNs
        self.assertEqual(result.shape, data2048_2D.shape)
        self.assertTrue(np.isnan(result[3, 3]))

    def test3D(self):
        """Test the NaNs are inserted into 3D data"""
        # Make 3D data
        data256_3D = np.ones((4, 256, 2048))

        # Run the function
        result = u.nan_reference_pixels(data256_3D)

        # Check shape and NaNs
        self.assertEqual(result.shape, data256_3D.shape)
        self.assertTrue(np.isnan(result[0, 3, 3]))

    def test4D(self):
        """Test the NaNs are inserted into 4D data"""
        # Make 4D data
        data96_4D = np.ones((2, 2, 96, 2048))

        # Run the function
        result = u.nan_reference_pixels(data96_4D)

        # Check shape and NaNs
        self.assertEqual(result.shape, data96_4D.shape)
        self.assertTrue(np.isnan(result[0, 0, 3, 3]))

    def testRaise(self):
        # Bad data
        bad_shape = np.ones((1, 2, 3, 96, 2048))

        # Throw error if 5+ dimensions
        self.assertRaises(ValueError, u.nan_reference_pixels, bad_shape)

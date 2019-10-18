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

    def test_init(self):
        """Test that the spectra can be combined"""
        # Run function
        s3 = u.combine_spectra(self.s1, self.s2)

        # Check combined spectra result
        self.assertEqual(s3.ndim, 2)
        self.assertEqual(s3.shape[0], 3)

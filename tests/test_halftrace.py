#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `halftrace` module."""

import unittest
from pkg_resources import resource_filename

from specialsoss import halftrace as ht


class TestHalftrace(unittest.TestCase):
    """Test functions in halftrace.py"""
    def setUp(self):
        """Test instance setup"""
        # Make dummy data
        self.tso4d = np.ones((2, 2, 256, 2048))

    def test_extract(self):
        """Test for extract function"""
        # Filters and subarays
        filters = 'CLEAR', 'F277W'
        subarrays = 'SUBSTRIP256', 'SUBSTRIP96', 'FULL'

        for filt in filters:
            for subarray in subarrays:

                # Run the extraction
                result = ht.extract(self.tso4d, filt=filt, subarray=subarray)


def test_halfmask():
    """Test the halfmask function"""
    # SUBSTRIP96
    sub96 = halftrace.halfmasks(subarray='SUBSTRIP96', plot=True)
    assert sub96[0].shape == sub96[1].shape == (96, 2048)

    # SUBSTRIP256
    sub96 = halftrace.halfmasks(subarray='SUBSTRIP256')
    assert sub96[0].shape == sub96[1].shape == (256, 2048)

    # FULL
    sub96 = halftrace.halfmasks(subarray='FULL')
    assert sub96[0].shape == sub96[1].shape == (2048, 2048)
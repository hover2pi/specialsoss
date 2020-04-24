#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `halftrace` module."""

import unittest
from pkg_resources import resource_filename

from specialsoss import halftrace


class TestHalftrace(unittest.TestCase):
    """Test functions in halftrace.py"""
    def setUp(self):
        """Test instance setup"""
        # Get files for testing
        self.frame = np.ones((256, 2048))
        self.tso3d = np.ones((4, 256, 2048))
        self.tso4d = np.ones((2, 2, 256, 2048))


def test_halfmask():
    """Test the halfmask function"""
    # SUBSTRIP96
    sub96 = halftrace.halfmasks(subarray='SUBSTRIP96', plot=True)
    assert sub96[0].shape == sub96[1].shape == (2048, 96)

    # SUBSTRIP256
    sub96 = halftrace.halfmasks(subarray='SUBSTRIP256')
    assert sub96[0].shape == sub96[1].shape == (2048, 256)

    # FULL
    sub96 = halftrace.halfmasks(subarray='FULL')
    assert sub96[0].shape == sub96[1].shape == (2048, 2048)
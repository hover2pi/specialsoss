#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `specialsoss` package."""

import pytest
from pkg_resources import resource_filename

from specialsoss import specialsoss


def test_SossObs():
    """Test SossObs object"""
    file = resource_filename('specialsoss', 'files/soss_sample.fits')
    obs = specialsoss.SossObs(filepath=file)
    assert obs.name == 'My SOSS Observations'
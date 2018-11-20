#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `specialsoss` package."""

import pytest

from specialsoss import specialsoss


def test_SossObs():
    """Test SossObs object"""
    obs = specialsoss.SossObs(path='.')
    assert obs.name == 'My SOSS Observations'
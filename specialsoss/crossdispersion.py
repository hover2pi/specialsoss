# -*- coding: utf-8 -*-
"""A module of functions that can be use to approximate the SOSS psf in the
cross dispersion direction"""

import os

import numpy as np


def batman(x, mu1, sigma1, A1, sigma2, A2, sep):
    """
    Generate a batman function of the given parameters

    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the function
    mu1: float
        The x-position of the first peak center
    sigma1: float
        The stanfard deviation of the two peak distributions
    A1: float
        The amplitude of the two peaks
    sigma2: float
        The stanfard deviation of the center peak
    A2: float
        The amplitude of the center peak
    sep: float
        The separation between the two peaks

    Returns
    -------
    np.ndarray
        The y-axis values of the mixed Gaussian
    """
    peak1 = gaussian(x, mu1, sigma1, A1)
    peak2 = gaussian(x, mu1+(sep/2.), sigma2, A2)
    peak3 = gaussian(x, mu1+sep, sigma1, A1)

    return peak1 + peak2 + peak3


def batmen(x, mu1, sigma1, A1, sigma2, A2, sep, mu3, sigma3, A3, sigma4, A4):
    """
    Generate two batman functions of the given parameters

    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the function
    mu1: float
        The x-position of the first peak center
    sigma1: float
        The stanfard deviation of the two peak distributions
    A1: float
        The amplitude of the two peaks
    sigma2: float
        The stanfard deviation of the center peak
    A2: float
        The amplitude of the center peak
    sep: float
        The separation between the two peaks

    Returns
    -------
    np.ndarray
        The y-axis values of the mixed Gaussian
    """
    batman1 = batman(x, mu1, sigma1, A1, sigma2, A2, sep)
    batman2 = batman(x, mu3, sigma3, A3, sigma4, A4, sep)

    return batman1 + batman2


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    """
    Generate a bimodal function of two Gaussians of the given parameters

    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the function
    mu1: float
        The x-position of the first peak center
    sigma1: float
        The stanfard deviation of the first distribution
    A1: float
        The amplitude of the first peak
    mu2: float
        The x-position of the second peak center
    sigma2: float
        The stanfard deviation of the second distribution
    A2: float
        The amplitude of the second peak

    Returns
    -------
    np.ndarray
        The y-axis values of the mixed Gaussian
    """
    g1 = gaussian(x, mu1, sigma1, A1)
    g2 = gaussian(x, mu2, sigma2, A2)

    return g1 + g2


def gaussian(x, mu, sigma, A):
    """
    Generate a Gaussian function of the given parameters

    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the Gaussian
    mu: float
        The x-position of the peak center
    sigma: float
        The stanfard deviation of the distribution
    A: float
        The amplitude of the peak

    Returns
    -------
    np.ndarray
        The y-axis values of the Gaussian
    """
    return A*np.exp(-(x-mu)**2/2/sigma**2)


def trimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
    """
    Generate a trimodal function of three Gaussians of the given parameters

    Parameters
    ----------
    x: array-like
        The x-axis on which to generate the function
    mu1: float
        The x-position of the first peak center
    sigma1: float
        The stanfard deviation of the first distribution
    A1: float
        The amplitude of the first peak
    mu2: float
        The x-position of the second peak center
    sigma2: float
        The stanfard deviation of the second distribution
    A2: float
        The amplitude of the second peak
    mu3: float
        The x-position of the third peak center
    sigma3: float
        The stanfard deviation of the third distribution
    A3: float
        The amplitude of the third peak

    Returns
    -------
    np.ndarray
        The y-axis values of the mixed Gaussian
    """
    g1 = gaussian(x, mu1, sigma1, A1)
    g2 = gaussian(x, mu2, sigma2, A2)
    g3 = gaussian(x, mu3, sigma3, A3)

    return g1 + g2 + g3

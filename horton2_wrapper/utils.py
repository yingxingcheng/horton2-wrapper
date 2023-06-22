from __future__ import division, print_function

import logging
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = [
    "get_gauss_legendre_points_lambda",
    "freq_grid",
    "integrate",
    "get_logger",
]

try:
    from opt_einsum import contract
except ImportError:
    contract = np.einsum


def freq_grid(n, L=0.3):
    """
    Gaussian-Legendre quadrature points and weights.
    Args:
        n: the number of points generated.
        L: scale, in libmbd, this value is 0.6 default, but here, we take 0.3.
    Returns:
        A tuple that contains points and weights.
    """
    x, w = leggauss(n)
    w = 2 * L / (1 - x) ** 2 * w
    x = L * (1 + x) / (1 - x)
    return x, w


def get_gauss_legendre_points_lambda(nw=16):
    """Gauss-Legendre points and weights."""
    pts, weights = leggauss(nw)
    pts = pts.real
    new_pts = 1 / 2 * pts + 1 / 2
    return new_pts, weights / 2


def integrate(*args):
    """Multiple matrices product."""
    return np.linalg.multi_dot(args)


def get_logger(
    name,
    level=logging.DEBUG,
    log_format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
):
    """Get logger for info printing."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(log_format)
    sh = logging.StreamHandler(stream=stream)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

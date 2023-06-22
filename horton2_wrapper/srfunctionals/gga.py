from __future__ import division, print_function

import numpy as np
from PBE_ERFGWS_correlation import d1esrc_pbe_gws_erf, d2esrc_pbe_gws_erf, esrc_pbe_gws_erf
from PBE_ERFGWS_exchange import (
    d1esrx_pbe_gws_erf_case_1,
    d1esrx_pbe_gws_erf_case_2_1,
    d1esrx_pbe_gws_erf_case_2_2,
    d1esrx_pbe_gws_erf_case_2_3,
    d1esrx_pbe_gws_erf_case_3,
    d2esrx_pbe_gws_erf_case_1,
    d2esrx_pbe_gws_erf_case_2_1,
    d2esrx_pbe_gws_erf_case_2_2,
    d2esrx_pbe_gws_erf_case_2_3,
    d2esrx_pbe_gws_erf_case_3,
    esrx_pbe_gws_erf_case_1,
    esrx_pbe_gws_erf_case_2_1,
    esrx_pbe_gws_erf_case_2_2,
    esrx_pbe_gws_erf_case_2_3,
    esrx_pbe_gws_erf_case_3,
)

__all__ = ["xc_gga_exc", "xc_gga_vxc", "xc_gga_fxc", "pbe_x_gwserf_unpolar", "pbe_c_gwserf_unpolar"]


def pbe_x_gwserf_unpolar(rho, sigma, mu, diff_order):
    """
    GWS version short-range PBE functional for unpolarized density.

    Parameters
    ----------
    rho: float
        Density
    sigma: float
        The norm square of the gradient of the density.
    mu: float
        The parameter used in range reparation.
    diff_order: int
        The order of differentiation of energy.

    Returns
    -------

    """
    # zk, vrho, vsigma, v2rho2, v2rhosigma, v2rhosigma, v2sigma2
    res = np.zeros(6, dtype=float)

    if rho <= 1e-10:
        if diff_order == 0:
            return res[0]
        elif diff_order == 1:
            return res[0], res[1:3]
        elif diff_order == 2:
            return res[0], res[1:3], res[3:]
        else:
            raise TypeError("type error")

    kF = rho ** (1 / 3) * ((3 * np.pi**2) ** (1 / 3))
    A = mu / (2 * kF)

    if diff_order == 0:
        if A < 1e-9:
            return esrx_pbe_gws_erf_case_1(rho, sigma, mu)
        elif A < 0.075:
            return esrx_pbe_gws_erf_case_2_1(rho, sigma, mu)
        elif A < 50:
            return esrx_pbe_gws_erf_case_2_3(rho, sigma, mu)
        elif A < 1e2:
            return esrx_pbe_gws_erf_case_2_2(rho, sigma, mu)
        elif A < 1e9:
            return esrx_pbe_gws_erf_case_3(rho, sigma, mu)
        return res[0]

    elif diff_order == 1:
        if A < 1e-9:
            return d1esrx_pbe_gws_erf_case_1(rho, sigma, mu)
        elif A < 0.075:
            return d1esrx_pbe_gws_erf_case_2_1(rho, sigma, mu)
        elif A < 50:
            return d1esrx_pbe_gws_erf_case_2_3(rho, sigma, mu)
        elif A < 1e2:
            return d1esrx_pbe_gws_erf_case_2_2(rho, sigma, mu)
        elif A < 1e9:
            return d1esrx_pbe_gws_erf_case_3(rho, sigma, mu)
        return res[0], res[1:3]

    elif diff_order == 2:
        if A < 1e-9:
            return d2esrx_pbe_gws_erf_case_1(rho, sigma, mu)
        elif A < 0.075:
            return d2esrx_pbe_gws_erf_case_2_1(rho, sigma, mu)
        elif A < 50:
            return d2esrx_pbe_gws_erf_case_2_3(rho, sigma, mu)
        elif A < 1e2:
            return d2esrx_pbe_gws_erf_case_2_2(rho, sigma, mu)
        elif A < 1e9:
            return d2esrx_pbe_gws_erf_case_3(rho, sigma, mu)
        else:
            return res[0], res[1:3], res[3:]
    else:
        raise TypeError("diff_order can only be one of {0, 1, 2}!")


def pbe_c_gwserf_unpolar(rho, sigma, mu, diff_order):
    # zk, vrho, vsigma, v2rho2, v2rhosigma, v2rhosigma, v2sigma2
    res = np.zeros(6, dtype=float)

    if rho <= 1e-10:
        if diff_order == 0:
            return res[0]
        elif diff_order == 1:
            return res[0], res[1:3]
        elif diff_order == 2:
            return res[0], res[1:3], res[3:]
        else:
            raise TypeError("type error")

    if diff_order == 0:
        return esrc_pbe_gws_erf(rho, sigma, mu)
    elif diff_order == 1:
        return d1esrc_pbe_gws_erf(rho, sigma, mu)
    elif diff_order == 2:
        return d2esrc_pbe_gws_erf(rho, sigma, mu)
    else:
        raise TypeError("diff_order can only be one of {0, 1, 2}!")


def xc_gga(xc_name, rho, sigma, mu, diff_order):
    if xc_name == "sr_gga_x_pbe_erfgws":
        func = pbe_x_gwserf_unpolar
    elif xc_name == "sr_gga_c_pbe_erfgws":
        func = pbe_c_gwserf_unpolar
    else:
        raise TypeError("xc type error!")

    npt_rho = len(rho)
    npt_sigma = len(sigma)
    assert npt_sigma == npt_rho

    zk = np.zeros((npt_rho, 1), dtype=float)
    if diff_order > 0:
        vrho = np.zeros((npt_rho, 1), dtype=float)
        vsigma = np.zeros((npt_rho, 1), dtype=float)

    if diff_order > 1:
        v2rho2 = np.zeros((npt_rho, 1), dtype=float)
        v2rhosigma = np.zeros((npt_rho, 1), dtype=float)
        v2sigma2 = np.zeros((npt_rho, 1), dtype=float)

    if diff_order == 0:
        for i in range(npt_rho):
            zk[i, 0] = func(rho[i], sigma[i], mu, 0)
        return {"zk": zk}
    elif diff_order == 1:
        for i in range(npt_rho):
            _zk, _d1e = func(rho[i], sigma[i], mu, 1)
            zk[i, 0] = _zk
            vrho[i, 0] = _d1e[0]
            vsigma[i, 0] = _d1e[1]
        return {"zk": zk, "vrho": vrho, "vsigma": vsigma}
    else:
        for i in range(npt_rho):
            _zk, _d1e, _d2e = func(rho[i], sigma[i], mu, 2)
            zk[i, 0] = _zk
            vrho[i, 0] = _d1e[0]
            vsigma[i, 0] = _d1e[1]
            v2rho2[i, 0] = _d2e[0]
            v2rhosigma[i, 0] = _d2e[1]
            v2sigma2[i, 0] = _d2e[2]
        return {
            "zk": zk,
            "vrho": vrho,
            "vsigma": vsigma,
            "v2rho2": v2rho2,
            "v2rhosigma": v2rhosigma,
            "v2sigma2": v2sigma2,
        }


def xc_gga_exc(xc_name, rho, sigma, mu):
    return xc_gga(xc_name, rho, sigma, mu, 0)


def xc_gga_vxc(xc_name, rho, sigma, mu):
    return xc_gga(xc_name, rho, sigma, mu, 1)


def xc_gga_fxc(xc_name, rho, sigma, mu):
    return xc_gga(xc_name, rho, sigma, mu, 2)

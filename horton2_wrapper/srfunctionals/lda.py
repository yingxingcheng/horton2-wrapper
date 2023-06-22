from __future__ import division, print_function

import numpy as np
from LDA_ERF_exchange import (
    d1esrx_lda_erf_case_1,
    d1esrx_lda_erf_case_2,
    d1esrx_lda_erf_case_3,
    d2esrx_lda_erf_case_1,
    d2esrx_lda_erf_case_2,
    d2esrx_lda_erf_case_3,
    esrx_lda_erf_case_1,
    esrx_lda_erf_case_2,
    esrx_lda_erf_case_3,
)
from PW92_ERF_correlation import d1esrc_pw92_erf, d2esrc_pw92_erf, esrc_pw92_erf
from VWN5_ERF_correlation import d1esrc_vwn5_erf, d2esrc_vwn5_erf, esrc_vwn5_erf

__all__ = [
    "lda_x_erf_unpolar",
    "lda_c_vwn5_erf_unpolar",
    "lda_c_pw92_erf_unpolar",
    "xc_lda",
    "xc_lda_exc",
    "xc_lda_fxc",
    "xc_lda_vxc",
]


def lda_x_erf_unpolar(rho, mu, diff_order):
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
            return esrx_lda_erf_case_1(rho, mu)
        elif A < 1e2:
            return esrx_lda_erf_case_2(rho, mu)
        elif A < 1e9:
            return esrx_lda_erf_case_3(rho, mu)
        return res[0]

    elif diff_order == 1:
        if A < 1e-9:
            return d1esrx_lda_erf_case_1(rho, mu)
        elif A < 1e2:
            return d1esrx_lda_erf_case_2(rho, mu)
        elif A < 1e9:
            return d1esrx_lda_erf_case_3(rho, mu)
        return res[0], res[1:3]

    elif diff_order == 2:
        if A < 1e-9:
            return d2esrx_lda_erf_case_1(rho, mu)
        elif A < 1e2:
            return d2esrx_lda_erf_case_2(rho, mu)
        elif A < 1e9:
            return d2esrx_lda_erf_case_3(rho, mu)
        else:
            return res[0], res[1:3], res[3:]
    else:
        raise TypeError("diff_order can only be one of {0, 1, 2}!")


def lda_c_pw92_erf_unpolar(rho, mu, diff_order):
    res = np.zeros(3, dtype=float)

    if rho <= 1e-10:
        if diff_order == 0:
            return res[0]
        elif diff_order == 1:
            return res[0], res[1:2]
        elif diff_order == 2:
            return res[0], res[1:2], res[2:]
        else:
            raise TypeError("type error")

    if diff_order == 0:
        return esrc_pw92_erf(rho, mu)
    elif diff_order == 1:
        return d1esrc_pw92_erf(rho, mu)
    elif diff_order == 2:
        return d2esrc_pw92_erf(rho, mu)
    else:
        raise TypeError("diff_order can only be one of {0, 1, 2}!")


def lda_c_vwn5_erf_unpolar(rho, mu, diff_order):
    res = np.zeros(3, dtype=float)

    if rho <= 1e-10:
        if diff_order == 0:
            return res[0]
        elif diff_order == 1:
            return res[0], res[1:2]
        elif diff_order == 2:
            return res[0], res[1:2], res[2:]
        else:
            raise TypeError("type error")

    if diff_order == 0:
        return esrc_vwn5_erf(rho, mu)
    elif diff_order == 1:
        return d1esrc_vwn5_erf(rho, mu)
    elif diff_order == 2:
        return d2esrc_vwn5_erf(rho, mu)
    else:
        raise TypeError("diff_order can only be one of {0, 1, 2}!")


def xc_lda(xc_name, rho, mu, diff_order):
    if xc_name == "sr_lda_x_erf":
        func = lda_x_erf_unpolar
    elif xc_name == "sr_lda_c_pw92_erf":
        func = lda_c_pw92_erf_unpolar
    elif xc_name == "sr_lda_c_vwn5_erf":
        func = lda_c_vwn5_erf_unpolar
    else:
        raise TypeError("xc type error!")

    npt_rho = len(rho)

    zk = np.zeros((npt_rho, 1), dtype=float)
    if diff_order > 0:
        vrho = np.zeros((npt_rho, 1), dtype=float)

    if diff_order > 1:
        v2rho2 = np.zeros((npt_rho, 1), dtype=float)

    if diff_order == 0:
        for i in range(npt_rho):
            zk[i, 0] = func(rho[i], mu, 0)
        return {"zk": zk}
    elif diff_order == 1:
        for i in range(npt_rho):
            _zk, _d1e = func(rho[i], mu, 1)
            zk[i, 0] = _zk
            vrho[i, 0] = _d1e[0]
        return {"zk": zk, "vrho": vrho}
    else:
        for i in range(npt_rho):
            _zk, _d1e, _d2e = func(rho[i], mu, 2)
            zk[i, 0] = _zk
            vrho[i, 0] = _d1e[0]
            v2rho2[i, 0] = _d2e[0]
        return {"zk": zk, "vrho": vrho, "v2rho2": v2rho2}


def xc_lda_exc(xc_name, rho, mu):
    return xc_lda(xc_name, rho, mu, 0)


def xc_lda_vxc(xc_name, rho, mu):
    return xc_lda(xc_name, rho, mu, 1)


def xc_lda_fxc(xc_name, rho, mu):
    return xc_lda(xc_name, rho, mu, 2)

from __future__ import division, print_function

import numpy as np
from TPSS_ERFGWS_correlation import d1esrc_tpss_gws_erf, d2esrc_tpss_gws_erf, esrc_tpss_gws_erf
from TPSS_ERFGWS_exchange import (
    d1esrx_tpss_gws_erf_case_1,
    d1esrx_tpss_gws_erf_case_2_1,
    d1esrx_tpss_gws_erf_case_2_2,
    d1esrx_tpss_gws_erf_case_2_3,
    d1esrx_tpss_gws_erf_case_3,
    d2esrx_tpss_gws_erf_case_1,
    d2esrx_tpss_gws_erf_case_2_1,
    d2esrx_tpss_gws_erf_case_2_2,
    d2esrx_tpss_gws_erf_case_2_3,
    d2esrx_tpss_gws_erf_case_3,
    esrx_tpss_gws_erf_case_1,
    esrx_tpss_gws_erf_case_2_1,
    esrx_tpss_gws_erf_case_2_2,
    esrx_tpss_gws_erf_case_2_3,
    esrx_tpss_gws_erf_case_3,
)


def tpss_x_gwserf_unpolar(rho, sigma, tau, lapl, mu, diff_order):
    # zk, vrho, vsigma, vtau, vlapl, v2rho2, v2rhosigma, v2rhosigma, v2sigma2
    res = np.zeros(15, dtype=float)

    if rho <= 1e-10 and sigma <= 1e-10 and tau <= 1e-10:
        if diff_order == 0:
            return res[0]
        elif diff_order == 1:
            return res[0], res[1:5]
        elif diff_order == 2:
            return res[0], res[1:5], res[5:]
        else:
            raise TypeError("type error")

    kF = rho ** (1 / 3) * ((3 * np.pi**2) ** (1 / 3))
    A = mu / (2 * kF)

    if diff_order == 0:
        if A < 1e-9:
            return esrx_tpss_gws_erf_case_1(rho, sigma, tau, lapl, mu)
        elif A < 0.075:
            return esrx_tpss_gws_erf_case_2_1(rho, sigma, tau, lapl, mu)
        elif A < 50:
            return esrx_tpss_gws_erf_case_2_3(rho, sigma, tau, lapl, mu)
        elif A < 1e2:
            return esrx_tpss_gws_erf_case_2_2(rho, sigma, tau, lapl, mu)
        elif A < 1e9:
            return esrx_tpss_gws_erf_case_3(rho, sigma, tau, lapl, mu)
        return res[0]

    elif diff_order == 1:
        if A < 1e-9:
            return d1esrx_tpss_gws_erf_case_1(rho, sigma, tau, lapl, mu)
        elif A < 0.075:
            return d1esrx_tpss_gws_erf_case_2_1(rho, sigma, tau, lapl, mu)
        elif A < 50:
            return d1esrx_tpss_gws_erf_case_2_3(rho, sigma, tau, lapl, mu)
        elif A < 1e2:
            return d1esrx_tpss_gws_erf_case_2_2(rho, sigma, tau, lapl, mu)
        elif A < 1e9:
            return d1esrx_tpss_gws_erf_case_3(rho, sigma, tau, lapl, mu)
        return res[0], res[1:5]

    elif diff_order == 2:
        if A < 1e-9:
            return d2esrx_tpss_gws_erf_case_1(rho, sigma, tau, lapl, mu)
        elif A < 0.075:
            return d2esrx_tpss_gws_erf_case_2_1(rho, sigma, tau, lapl, mu)
        elif A < 50:
            return d2esrx_tpss_gws_erf_case_2_3(rho, sigma, tau, lapl, mu)
        elif A < 1e2:
            return d2esrx_tpss_gws_erf_case_2_2(rho, sigma, tau, lapl, mu)
        elif A < 1e9:
            return d2esrx_tpss_gws_erf_case_3(rho, sigma, tau, lapl, mu)
        else:
            return res[0], res[1:5], res[5:]
    else:
        raise TypeError("diff_order can only be one of {0, 1, 2}!")


def tpss_c_gwserf_unpolar(rho, sigma, tau, lapl, mu, diff_order):
    # zk, vrho, vsigma, vtau, vlapl
    res = np.zeros(15, dtype=float)

    if rho <= 1e-10:
        if diff_order == 0:
            return res[0]
        elif diff_order == 1:
            return res[0], res[1:5]
        elif diff_order == 2:
            return res[0], res[1:5], res[5:]
        else:
            raise TypeError("type error")

    if diff_order == 0:
        return esrc_tpss_gws_erf(rho, sigma, tau, lapl, mu)
    elif diff_order == 1:
        return d1esrc_tpss_gws_erf(rho, sigma, tau, lapl, mu)
    elif diff_order == 2:
        return d2esrc_tpss_gws_erf(rho, sigma, tau, lapl, mu)
    else:
        raise TypeError("diff_order can only be one of {0, 1, 2}!")


def xc_mgga(xc_name, rho, sigma, tau, lapl, mu, diff_order):
    if xc_name == "sr_mgga_x_tpss_erfgws":
        func = tpss_x_gwserf_unpolar
    elif xc_name == "sr_mgga_c_tpss_erfgws":
        func = tpss_c_gwserf_unpolar
    else:
        raise TypeError("xc type error!")

    npt_rho = len(rho)

    zk = np.zeros((npt_rho, 1), dtype=float)
    if diff_order > 0:
        vrho = np.zeros((npt_rho, 1), dtype=float)
        vsigma = np.zeros((npt_rho, 1), dtype=float)
        vtau = np.zeros((npt_rho, 1), dtype=float)
        vlapl = np.zeros((npt_rho, 1), dtype=float)

    if diff_order > 1:
        v2rho2 = np.zeros((npt_rho, 1), dtype=float)
        v2rhosigma = np.zeros((npt_rho, 1), dtype=float)
        v2sigma2 = np.zeros((npt_rho, 1), dtype=float)
        v2rhotau = np.zeros((npt_rho, 1), dtype=float)
        v2sigmatau = np.zeros((npt_rho, 1), dtype=float)
        v2tau2 = np.zeros((npt_rho, 1), dtype=float)
        v2rholapl = np.zeros((npt_rho, 1), dtype=float)
        v2sigmalapl = np.zeros((npt_rho, 1), dtype=float)
        v2taulapl = np.zeros((npt_rho, 1), dtype=float)
        v2lapl2 = np.zeros((npt_rho, 1), dtype=float)

    if diff_order == 0:
        for i in range(npt_rho):
            zk[i, 0] = func(rho[i], sigma[i], tau[i], lapl[i], mu, 0)
        return {"zk": zk}
    elif diff_order == 1:
        for i in range(npt_rho):
            _zk, _d1e = func(rho[i], sigma[i], tau[i], lapl[i], mu, 1)
            zk[i, 0] = _zk
            vrho[i, 0] = _d1e[0]
            vsigma[i, 0] = _d1e[1]
            vtau[i, 0] = _d1e[2]
            vlapl[i, 0] = _d1e[3]
        return {"zk": zk, "vrho": vrho, "vsigma": vsigma, "vtau": vtau, "vlapl": vlapl}
    else:
        for i in range(npt_rho):
            _zk, _d1e, _d2e = func(rho[i], sigma[i], tau[i], lapl[i], mu, 2)
            zk[i, 0] = _zk
            vrho[i, 0] = _d1e[0]
            vsigma[i, 0] = _d1e[1]
            vtau[i, 0] = _d1e[2]
            vlapl[i, 0] = _d1e[3]

            v2rho2[i, 0] = _d2e[0]
            v2rhosigma[i, 0] = _d2e[1]
            v2sigma2[i, 0] = _d2e[2]
            v2rhotau[i, 0] = _d2e[3]
            v2sigmatau[i, 0] = _d2e[4]
            v2tau2[i, 0] = _d2e[5]
            v2rholapl[i, 0] = _d2e[6]
            v2sigmalapl[i, 0] = _d2e[7]
            v2taulapl[i, 0] = _d2e[8]
            v2lapl2[i, 0] = _d2e[9]

        return {
            "zk": zk,
            "vrho": vrho,
            "vsigma": vsigma,
            "vtau": vtau,
            "vlapl": vlapl,
            "v2rho2": v2rho2,
            "v2rhosigma": v2rhosigma,
            "v2sigma2": v2sigma2,
            "v2rhotau": v2rhotau,
            "v2sigmatau": v2sigmatau,
            "v2tau2": v2tau2,
            "v2rholapl": v2rholapl,
            "v2sigmalapl": v2sigmalapl,
            "v2taulapl": v2taulapl,
            "v2lapl2": v2lapl2,
        }


def xc_mgga_exc(xc_name, rho, sigma, tau, lapl, mu):
    return xc_mgga(xc_name, rho, sigma, tau, lapl, mu, 0)


def xc_mgga_vxc(xc_name, rho, sigma, tau, lapl, mu):
    return xc_mgga(xc_name, rho, sigma, tau, lapl, mu, 1)


def xc_mgga_fxc(xc_name, rho, sigma, tau, lapl, mu):
    return xc_mgga(xc_name, rho, sigma, tau, lapl, mu, 2)

from __future__ import division, print_function

import numpy as np
import pylibxc
import pytest

from .mgga import xc_mgga_exc, xc_mgga_fxc, xc_mgga_vxc


def test_pbe_x_erfgws_exc():
    rho = np.array([1, 2, 3, 4])
    # sigma = np.array([2, 3, 4, 5])
    sigma = np.array([20, 13, 14, 25])
    tau = np.array([2, 3, 4, 5])
    lapl = np.zeros((4,))

    res = xc_mgga_exc("sr_mgga_x_tpss_erfgws", rho, sigma, tau, lapl, 0.0)
    print(res)

    inp = {"rho": rho, "sigma": sigma, "tau": tau, "lapl": lapl}
    func = pylibxc.LibXCFunctional("mgga_x_tpss", "unpolarized")
    ret = func.compute(inp, do_vxc=False)
    print(ret)

    for k, v in res.items():
        assert v == pytest.approx(ret[k], abs=1e-5)


def test_pbe_x_erfgws_vxc():
    rho = np.array([1, 2, 3, 4])
    # sigma = np.array([2, 3, 4, 5])
    sigma = np.array([20, 13, 14, 25])
    tau = np.array([2, 3, 4, 5])
    lapl = np.zeros((4,))

    res = xc_mgga_vxc("sr_mgga_x_tpss_erfgws", rho, sigma, tau, lapl, 0.0)
    print(res)

    inp = {"rho": rho, "sigma": sigma, "tau": tau, "lapl": lapl}
    func = pylibxc.LibXCFunctional("mgga_x_tpss", "unpolarized")
    ret = func.compute(inp)
    print(ret)

    for k, v in res.items():
        assert v == pytest.approx(ret[k], abs=1e-5)


def test_pbe_c_erfgws_vxc():
    rho = np.array([1, 2, 3, 4])
    # sigma = np.array([2, 3, 4, 5])
    sigma = np.array([20, 13, 14, 25])
    tau = np.array([2, 3, 4, 5])
    lapl = np.zeros((4,))
    res = xc_mgga_vxc("sr_mgga_c_tpss_erfgws", rho, sigma, tau, lapl, 0.0)
    print(res)

    inp = {"rho": rho, "sigma": sigma, "tau": tau, "lapl": lapl}
    func = pylibxc.LibXCFunctional("mgga_c_tpss", "unpolarized")
    ret = func.compute(inp)
    print(ret)

    for k, v in res.items():
        assert v == pytest.approx(ret[k], abs=1e-5)


def test_pbe_x_erfgws_fxc():
    rho = np.array([1, 2, 3, 4])
    # sigma = np.array([2, 3, 4, 5])
    sigma = np.array([20, 13, 14, 25])
    tau = np.array([2, 3, 4, 5])
    lapl = np.zeros((4,))

    res = xc_mgga_fxc("sr_mgga_x_tpss_erfgws", rho, sigma, tau, lapl, 0.0)
    print(res)

    inp = {"rho": rho, "sigma": sigma, "tau": tau, "lapl": lapl}
    func = pylibxc.LibXCFunctional("mgga_x_tpss", "unpolarized")
    ret = func.compute(inp, do_fxc=True)
    print(ret)

    for k, v in res.items():
        assert v == pytest.approx(ret[k], abs=1e-5)

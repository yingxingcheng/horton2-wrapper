from __future__ import division, print_function

import numpy as np
import pylibxc

from horton2_wrapper.srfunctionals.gga import xc_gga_fxc


def test_pbe_x_erfgws_fxc():
    rho = np.array([1, 2, 3, 4])
    sigma = np.array([2, 3, 4, 5])
    res = xc_gga_fxc("sr_gga_x_pbe_erfgws", rho, sigma, 0.0)
    print(res)

    inp = {"rho": rho, "sigma": sigma}
    func = pylibxc.LibXCFunctional("gga_x_pbe", "unpolarized")
    ret = func.compute(inp, do_fxc=True)
    print(ret)

    for k, v in res.items():
        assert np.allclose(v, ret[k])


def test_pbe_c_erfgws_fxc():
    rho = np.array([1, 2, 3, 4])
    sigma = np.array([2, 3, 4, 5])
    res = xc_gga_fxc("sr_gga_c_pbe_erfgws", rho, sigma, 0.0)
    print(res)

    inp = {"rho": rho, "sigma": sigma}
    func = pylibxc.LibXCFunctional("gga_c_pbe", "unpolarized")
    ret = func.compute(inp, do_fxc=True)
    print(ret)

    for k, v in res.items():
        assert np.allclose(v, ret[k])

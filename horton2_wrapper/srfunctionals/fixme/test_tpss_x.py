from __future__ import division, print_function

import numpy as np
import pytest
from TPSS_nomu_exchange import d1esrx_tpss, d2esrx_tpss, esrx_tpss

from .tool import f2libxc, get_mgga_ref


def test_zk_unpolar():
    rho = 1
    sigma = 20
    tau = 2
    lapl = 0.0
    zk = esrx_tpss(rho, sigma, tau, lapl)
    res = f2libxc(zk, spin="unpolarized", xc_type="mgga")
    res_ref = get_mgga_ref(
        "mgga_x_tpss", "unpolarized", rho, 0, sigma, 0, 0, tau, 0, lapl, 0, do_vxc=False
    )
    print(res)
    print(res_ref)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())

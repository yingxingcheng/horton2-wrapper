import numpy as np
import pytest

from lrccd.srfunctionals.PBE_ERFGWS_exchange import (
    d1esrx_pbe_gws_erf_case_1,
    d2esrx_pbe_gws_erf_case_1,
    esrx_pbe_gws_erf_case_1,
)
from lrccd.srfunctionals.tool import f2libxc, get_gga_ref


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 20))
@pytest.mark.parametrize("gamma", np.random.uniform(1e-10, 1e5, 20))
def test_zk_unpolar(rho, gamma):
    zk = esrx_pbe_gws_erf_case_1(rho, gamma, 0.0)
    res = f2libxc(zk, spin="unpolarized", xc_type="gga")
    res_ref = get_gga_ref("gga_x_pbe", "unpolarized", rho, 0.0, gamma, 0.0, 0.0, do_vxc=False)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())


def test_zk_unpolar_zero():
    rho, gamma = 0, 0
    zk = esrx_pbe_gws_erf_case_1(rho, gamma, 0.0)
    res = f2libxc(zk, spin="unpolarized", xc_type="gga")
    for k, v in res.items():
        assert np.isnan(v.flatten()[0])


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 20))
@pytest.mark.parametrize("gamma", np.random.uniform(1e-10, 1e5, 20))
def test_d1e_unpolar(rho, gamma):
    zk, d1e = d1esrx_pbe_gws_erf_case_1(rho, gamma, 0.0)
    res = f2libxc(zk, d1e, spin="unpolarized", xc_type="gga")
    res_ref = get_gga_ref("gga_x_pbe", "unpolarized", rho, 0.0, gamma, 0.0, 0.0)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 20))
@pytest.mark.parametrize("gamma", np.random.uniform(1e-10, 1e5, 20))
def test_d2e_unpolar(rho, gamma):
    zk, d1e, d2e = d2esrx_pbe_gws_erf_case_1(rho, gamma, 0.0)
    res = f2libxc(zk, d1e, d2e, spin="unpolarized", xc_type="gga")
    res_ref = get_gga_ref("gga_x_pbe", "unpolarized", rho, 0.0, gamma, 0.0, 0.0, do_fxc=True)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())

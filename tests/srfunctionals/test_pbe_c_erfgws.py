import numpy as np
import pytest

from lrccd.srfunctionals.PBE_ERFGWS_correlation import (
    d1esrc_pbe_gws_erf,
    d1esrc_spin_pbe_gws_erf,
    d2esrc_pbe_gws_erf,
    d2esrc_spin_pbe_gws_erf,
    esrc_pbe_gws_erf,
    esrc_spin_pbe_gws_erf,
)
from lrccd.srfunctionals.tool import f2libxc, get_gga_ref


@pytest.mark.parametrize("rho", np.random.uniform(0.01, 1e5, 20))
@pytest.mark.parametrize("gamma", np.random.uniform(0.01, 1e5, 20))
def test_zk_unpolar(rho, gamma):
    zk = esrc_pbe_gws_erf(rho, gamma, 0.0)
    res = f2libxc(zk, spin="unpolarized", xc_type="gga")
    res_ref = get_gga_ref("gga_c_pbe", "unpolarized", rho, 0.0, gamma, 0.0, 0.0, do_vxc=False)
    for k, v in res.items():
        assert v == pytest.approx(res_ref[k].flatten(), abs=1e-6)


@pytest.mark.parametrize("rho0", np.random.uniform(0.01, 1e5, 3))
@pytest.mark.parametrize("rho1", np.random.uniform(0.01, 1e5, 3))
@pytest.mark.parametrize("gamma0", np.random.uniform(20, 100, 3))
@pytest.mark.parametrize("gamma1", np.random.uniform(0.01, 1e5, 3))
@pytest.mark.parametrize("gamma2", np.random.uniform(20, 100, 3))
def test_zk_polar(rho0, rho1, gamma0, gamma1, gamma2):
    # Note: the range of rho0 and rho1 are very important, since zeeta = (rho0 - rho1)/(rho0 + rho1)
    # cannot be too small value due to zeta**(-n) where n is positive real number.

    # gamma0 + gamma2 >= 2 * gamma1
    zk = esrc_spin_pbe_gws_erf(rho0, rho1, gamma0, gamma1, gamma2, 0.0)
    res = f2libxc(zk, spin="polarized", xc_type="gga")
    res_ref = get_gga_ref(
        "gga_c_pbe", "polarized", rho0, rho1, gamma0, gamma1, gamma2, do_vxc=False
    )
    for k, v in res.items():
        assert v == pytest.approx(res_ref[k].flatten(), abs=1e-6)


@pytest.mark.parametrize("rho", np.random.uniform(0.01, 1e5, 20))
@pytest.mark.parametrize("gamma", np.random.uniform(0.01, 1e5, 20))
def test_d1e_unpolar(rho, gamma):
    zk, d1e = d1esrc_pbe_gws_erf(rho, gamma, 0.0)
    res = f2libxc(zk, d1e, spin="unpolarized", xc_type="gga")
    res_ref = get_gga_ref("gga_c_pbe", "unpolarized", rho, 0.0, gamma, 0.0, 0.0)
    for k, v in res.items():
        assert v == pytest.approx(res_ref[k].flatten(), abs=1e-6)


@pytest.mark.parametrize("rho0", np.random.uniform(0.01, 50, 3))
@pytest.mark.parametrize("rho1", np.random.uniform(0.01, 50, 3))
@pytest.mark.parametrize("gamma0", np.random.uniform(20, 100, 3))
@pytest.mark.parametrize("gamma1", np.random.uniform(0.01, 20, 3))
@pytest.mark.parametrize("gamma2", np.random.uniform(20, 100, 3))
def test_d1e_polar(rho0, rho1, gamma0, gamma1, gamma2):
    # Note: the range of rho0 and rho1 are very important, since zeeta = (rho0 - rho1)/(rho0 + rho1)
    # cannot be too small value due to zeta**(-n) where n is positive real number.

    # gamma0 + gamma2 >= 2 * gamma1
    zk, d1e = d1esrc_spin_pbe_gws_erf(rho0, rho1, gamma0, gamma1, gamma2, 0.0)
    res = f2libxc(zk, d1e, spin="polarized", xc_type="gga")
    res_ref = get_gga_ref("gga_c_pbe", "polarized", rho0, rho1, gamma0, gamma1, gamma2)
    for k, v in res.items():
        assert v == pytest.approx(res_ref[k].flatten(), abs=1e-6)


@pytest.mark.parametrize("rho", np.random.uniform(0.01, 1e5, 20))
@pytest.mark.parametrize("gamma", np.random.uniform(0.01, 1e5, 20))
def test_d2e_unpolar(rho, gamma):
    zk, d1e, d2e = d2esrc_pbe_gws_erf(rho, gamma, 0.0)
    res = f2libxc(zk, d1e, d2e, spin="unpolarized", xc_type="gga")
    res_ref = get_gga_ref("gga_c_pbe", "unpolarized", rho, 0.0, gamma, 0.0, 0.0, do_fxc=True)
    for k, v in res.items():
        assert v == pytest.approx(res_ref[k].flatten(), abs=1e-6)


@pytest.mark.parametrize("rho0", np.random.uniform(0.01, 50, 3))
@pytest.mark.parametrize("rho1", np.random.uniform(0.01, 50, 3))
@pytest.mark.parametrize("gamma0", np.random.uniform(20, 100, 3))
@pytest.mark.parametrize("gamma1", np.random.uniform(0.01, 20, 3))
@pytest.mark.parametrize("gamma2", np.random.uniform(20, 100, 3))
def test_d2e_polar(rho0, rho1, gamma0, gamma1, gamma2):
    zk, d1e, d2e = d2esrc_spin_pbe_gws_erf(rho0, rho1, gamma0, gamma1, gamma2, 0.0)
    res = f2libxc(zk, d1e, d2e, spin="polarized", xc_type="gga")
    res_ref = get_gga_ref("gga_c_pbe", "polarized", rho0, rho1, gamma0, gamma1, gamma2, do_fxc=True)
    for k, v in res.items():
        assert v == pytest.approx(res_ref[k].flatten(), abs=1e-6)

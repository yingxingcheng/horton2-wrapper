import numpy as np
import pytest

from lrccd.srfunctionals.tool import f2libxc, get_lda_ref
from lrccd.srfunctionals.VWN5_nomu_correlation import (
    d1esrc_spin_vwn5,
    d1esrc_vwn5,
    d2esrc_spin_vwn5,
    d2esrc_vwn5,
    esrc_spin_vwn5,
    esrc_vwn5,
)


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 400))
def test_zk_unpolar(rho):
    zk = esrc_vwn5(rho)
    res = f2libxc(zk, spin="unpolarized", xc_type="lda")
    res_ref = get_lda_ref("lda_c_vwn", "unpolarized", rho, 0.0, do_vxc=False)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())


@pytest.mark.parametrize("rho0", np.random.uniform(1e-10, 50, 20))
@pytest.mark.parametrize("rho1", np.random.uniform(1e-10, 50, 20))
def test_zk_polar(rho0, rho1):
    # Note: the range of rho0 and rho1 are very important, since zeeta = (rho0 - rho1)/(rho0 + rho1)
    # cannot be too small value due to zeta**(-n) where n is positive real number.
    zk = esrc_spin_vwn5(rho0, rho1)
    res = f2libxc(zk, spin="polarized", xc_type="lda")
    res_ref = get_lda_ref("lda_c_vwn", "polarized", rho0, rho1, do_vxc=False)
    for k, v in res.items():
        assert v == pytest.approx(res_ref[k].flatten(), abs=1e-4)


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 400))
def test_d1e_unpolar(rho):
    # TODO: tho cannot be zero
    rho = rho / 1e3
    zk, d1e = d1esrc_vwn5(rho)
    res = f2libxc(zk, d1e, spin="unpolarized", xc_type="lda")
    res_ref = get_lda_ref("lda_c_vwn", "unpolarized", rho, 0.0)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())


def test_esr_unpolar_zero():
    rho = 0
    zk = esrc_vwn5(rho)
    res = f2libxc(zk, spin="unpolarized", xc_type="lda")
    for k, v in res.items():
        assert np.isnan(v.flatten()[0])


def test_d1e_unpolar_zero():
    rho = 0
    zk, d1e = d1esrc_vwn5(rho)
    res = f2libxc(zk, d1e, spin="unpolarized", xc_type="lda")
    for k, v in res.items():
        assert np.isnan(v.flatten()[0])

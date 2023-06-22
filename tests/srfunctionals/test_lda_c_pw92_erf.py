import numpy as np
import pytest

from horton2_wrapper.srfunctionals.PW92_ERF_correlation import (
    d1esrc_pw92_erf,
    d2esrc_pw92_erf,
    esrc_pw92_erf,
)
from horton2_wrapper.srfunctionals.tool import f2libxc, get_lda_ref


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 400))
def test_zk_unpolar(rho):
    zk = esrc_pw92_erf(rho, 0.0)
    res = f2libxc(zk, spin="unpolarized", xc_type="lda")
    res_ref = get_lda_ref("lda_c_pw", "unpolarized", rho, 0.0, do_vxc=False)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 400))
def test_d1esrc_unpolar(rho):
    zk, d1e = d1esrc_pw92_erf(rho, 0.0)
    res = f2libxc(zk, d1e=d1e, spin="unpolarized", xc_type="lda")
    res_ref = get_lda_ref("lda_c_pw", "unpolarized", rho, 0.0)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 400))
def test_d2esrc_unpolar(rho):
    zk, d1e, d2e = d2esrc_pw92_erf(rho, 0.0)
    res = f2libxc(zk, d1e=d1e, d2e=d2e, spin="unpolarized", xc_type="lda")
    res_ref = get_lda_ref("lda_c_pw", "unpolarized", rho, 0.0, do_fxc=True)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())

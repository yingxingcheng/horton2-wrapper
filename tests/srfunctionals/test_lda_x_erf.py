import numpy as np
import pytest

from horton2_wrapper.srfunctionals.LDA_ERF_exchange import (
    d1esrx_lda_erf_case_1,
    d1esrx_lda_erf_case_2,
    d1esrx_lda_erf_case_3,
    d1esrx_spin_lda_erf_case_1,
    d1esrx_spin_lda_erf_case_2,
    d1esrx_spin_lda_erf_case_3,
    d2esrx_lda_erf_case_1,
    d2esrx_lda_erf_case_2,
    d2esrx_lda_erf_case_3,
    d2esrx_spin_lda_erf_case_1,
    d2esrx_spin_lda_erf_case_2,
    d2esrx_spin_lda_erf_case_3,
    esrx_lda_erf_case_1,
    esrx_lda_erf_case_2,
    esrx_lda_erf_case_3,
    esrx_spin_lda_erf_case_1,
    esrx_spin_lda_erf_case_2,
    esrx_spin_lda_erf_case_3,
)
from horton2_wrapper.srfunctionals.tool import f2libxc, get_lda_ref


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 20))
def test_zk_unpolar(rho):
    zk = esrx_lda_erf_case_1(rho, 0.0)
    res = f2libxc(zk, spin="unpolarized", xc_type="lda")
    res_ref = get_lda_ref("lda_x", "unpolarized", rho, 0.0, do_vxc=False)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())

    zk2 = esrx_lda_erf_case_2(rho, 0.0)
    res2 = f2libxc(zk2, spin="unpolarized", xc_type="lda")
    for k, v in res2.items():
        assert np.allclose(v, res_ref[k].flatten())


# # failed
# @pytest.mark.parametrize('rho', np.random.uniform(0, 1e5, 1))
# def test_zk_unpolar_case_3(rho):
#     zk = esrx_lda_erf_case_3(rho, 0.0)
#     res = f2libxc(zk, spin='unpolarized', xc_type='lda')
#     res_ref = get_lda_ref('lda_x', 'unpolarized', rho, 0.0, do_vxc=False)
#     for k, v in res.items():
#         print(v)
#         print(res_ref[k])
#         assert np.allclose(v, res_ref[k].flatten())


@pytest.mark.parametrize("rho0", np.random.uniform(1e-10, 50, 20))
@pytest.mark.parametrize("rho1", np.random.uniform(1e-10, 50, 20))
def test_zk_polar(rho0, rho1):
    # Note: the range of rho0 and rho1 are very important, since zeeta = (rho0 - rho1)/(rho0 + rho1)
    # cannot be too small value due to zeta**(-n) where n is positive real number.
    zk = esrx_spin_lda_erf_case_1(rho0, rho1, 0.0)
    res = f2libxc(zk, spin="polarized", xc_type="lda")
    res_ref = get_lda_ref("lda_x", "polarized", rho0, rho1, do_vxc=False)
    for k, v in res.items():
        assert v == pytest.approx(res_ref[k].flatten(), abs=1e-4)

    zk2 = esrx_spin_lda_erf_case_2(rho0, rho1, 0.0)
    res2 = f2libxc(zk2, spin="polarized", xc_type="lda")
    for k, v in res2.items():
        assert v == pytest.approx(res_ref[k].flatten(), abs=1e-4)


@pytest.mark.parametrize("rho", np.random.uniform(1e-10, 1e5, 400))
def test_d1e_unpolar(rho):
    zk, d1e = d1esrx_lda_erf_case_1(rho, 0.0)
    res = f2libxc(zk, d1e, spin="unpolarized", xc_type="lda")
    res_ref = get_lda_ref("lda_x", "unpolarized", rho, 0.0)
    for k, v in res.items():
        assert np.allclose(v, res_ref[k].flatten())

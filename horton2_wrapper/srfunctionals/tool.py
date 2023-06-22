import numpy as np
import pylibxc


def f2libxc(zk, d1e=None, d2e=None, spin="unpolarized", xc_type="lda"):
    res = {"zk": np.asarray([zk], dtype=float)}

    keys = ["rho"]
    if xc_type == "gga":
        keys = ["rho", "sigma"]
    elif xc_type == "mgga":
        keys = ["rho", "sigma", "tau", "lapl"]

    if spin == "unpolarized":
        nb_comps = {"rho": 1, "sigma": 1, "tau": 1, "lapl": 1}
    else:
        nb_comps = {"rho": 2, "sigma": 3, "tau": 2, "lapl": 2}

    if d1e is not None:
        begin = 0
        for i, k in enumerate(keys):
            res["v" + k] = d1e[begin : begin + nb_comps[k]]
            begin += nb_comps[k]

    if d2e is not None:
        begin = 0
        nb_key = len(keys)
        for i in range(nb_key):
            for j in range(i, nb_key):
                ki, kj = keys[i], keys[j]
                if ki == kj:
                    key = "v2{}2".format(ki)
                    nb_v2 = (1 + nb_comps[ki]) * nb_comps[ki] // 2
                else:
                    key = "v2{}{}".format(ki, kj)
                    nb_v2 = nb_comps[ki] * nb_comps[kj]

                res[key] = d2e[begin : begin + nb_v2]
                begin += nb_v2
    return res


def get_lda_ref(
    func_name, spin, rho0, rho1, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False
):
    func = pylibxc.LibXCFunctional(func_name, spin)
    if spin == "polarized":
        inp = {"rho": np.array([rho0, rho1])}
    else:
        inp = {"rho": np.array([rho0])}

    ret = func.compute(
        inp, do_exc=do_exc, do_vxc=do_vxc, do_fxc=do_fxc, do_kxc=do_kxc, do_lxc=do_lxc
    )
    return ret


def get_gga_ref(
    func_name,
    spin,
    rho0,
    rho1,
    gamma0,
    gamma1,
    gamma2,
    do_exc=True,
    do_vxc=True,
    do_fxc=False,
    do_kxc=False,
    do_lxc=False,
):
    func = pylibxc.LibXCFunctional(func_name, spin)
    if spin == "polarized":
        inp = {"rho": np.array([rho0, rho1]), "sigma": np.array([gamma0, gamma1, gamma2])}
    else:
        inp = {"rho": np.array([rho0]), "sigma": np.array([gamma0])}

    ret = func.compute(
        inp, do_exc=do_exc, do_vxc=do_vxc, do_fxc=do_fxc, do_kxc=do_kxc, do_lxc=do_lxc
    )
    return ret


def get_mgga_ref(
    func_name,
    spin,
    rho0,
    rho1,
    gamma0,
    gamma1,
    gamma2,
    tau0,
    tau1,
    lapl0,
    lapl1,
    do_exc=True,
    do_vxc=True,
    do_fxc=False,
    do_kxc=False,
    do_lxc=False,
):
    func = pylibxc.LibXCFunctional(func_name, spin)
    if spin == "polarized":
        inp = {
            "rho": np.array([rho0, rho1]),
            "sigma": np.array([gamma0, gamma1, gamma2]),
            "tau": np.array([tau0, tau1]),
            "lapl": np.array([lapl0, lapl1]),
        }
    else:
        inp = {
            "rho": np.array([rho0]),
            "sigma": np.array([gamma0]),
            "tau": np.array([tau0]),
            "lapl": np.array([lapl0]),
        }

    ret = func.compute(
        inp, do_exc=do_exc, do_vxc=do_vxc, do_fxc=do_fxc, do_kxc=do_kxc, do_lxc=do_lxc
    )
    return ret

import numpy as np
from horton import (
    RLibXCLDA,
    RLibXCGGA,
    RLibXCMGGA,
    RLibXCHybridGGA,
    RLibXCHybridMGGA,
    RLibXCWrapper,
    ULibXCWrapper,
    ULibXCMGGA,
    ULibXCLDA,
    ULibXCGGA,
    ULibXCHybridMGGA,
    ULibXCHybridGGA,
)

from horton2_wrapper.libsrxc import RLibXCSRLDA, RLibXCSRGGA, RLibXCSRMGGA, RLibXCSRWrapper

__all__ = [
    "XCHelper",
    "XC_DICT",
    "compute_gga_hardness_num",
    "compute_lda_hardness_num",
    "compute_gga_hardness",
    "compute_lda_hardness",
]

XC_DICT = {
    "lda": ["lda_x", "lda_c_vwn"],
    "lda_x": ["lda_x"],
    "pbe": ["gga_x_pbe", "gga_c_pbe"],
    "b3lyp": ["hyb_gga_xc_b3lyp"],
    "svwn5_rpa": ["lda_x", "lda_c_vwn_rpa"],
    "svwn1": ["lda_x", "lda_c_vwn_1"],
    "svwn2": ["lda_x", "lda_c_vwn_2"],
    "svwn3": ["lda_x", "lda_c_vwn_3"],
    "svwn4": ["lda_x", "lda_c_vwn_4"],
    "tpss": ["mgga_x_tpss", "mgga_c_tpss"],
    "pbe0_13": ["hyb_gga_xc_pbe0_13"],
    "m05": ["hyb_mgga_xc_m05"],
    # short-range functional
    "srlda": ["sr_lda_x_erf", "sr_lda_c_vwn5_erf"],
    "srpbe": ["sr_gga_x_pbe_erfgws", "sr_gga_c_pbe_erfgws"],
    "lrcpbe": ["sr_gga_x_pbe_erfgws", "gga_c_pbe"],
    "lrclda": ["sr_lda_x_erf", "lda_c_vwn"],
    "sr_ldax": ["sr_lda_x_erf"],
    "sr_lda_vwn5": ["sr_lda_x_erf", "sr_lda_c_vwn5_erf"],
    "sr_lda_pw92": ["sr_lda_x_erf", "sr_lda_c_pw92_erf"],
}
XC_DICT["svwn"] = XC_DICT["svwn5"] = XC_DICT["lda"]


class XCHelper(object):
    # family = {2: gga, 4: mgga, 32: hybridGGA, 64: hybridMGGA}

    def __init__(self, alise, spin=0, mu=0.4):
        if isinstance(alise, str):
            self.func_names = XC_DICT[str(alise).lower()]
        elif isinstance(alise, (list, tuple)):
            self.func_names = [str(xc_type).lower() for xc_type in alise]
        else:
            raise RuntimeError("The alise for xc functional is not correct: {}".format(alise))
        assert spin in [0, 1]
        self.spin = spin
        self.mu = mu

    def has_srxc(self):
        """Check whether a short-range functional is included."""
        tag = False
        for fn in self.func_names:
            if fn.startswith("sr_"):
                tag = True
                break
        return tag

    def get_xc_terms(self):
        """
        Build exchange-correlation functional terms defined Horton.

        Returns
        -------
        List:
            A list of exchange-correlation functional object.

        """
        objs = []
        for i, name in enumerate(self.func_names):
            if self.spin == 0:
                if name.startswith("sr_lda"):
                    objs.append(RLibXCSRLDA(name.replace("sr_lda_", ""), mu=self.mu))
                elif name.startswith("sr_gga"):
                    objs.append(RLibXCSRGGA(name.replace("sr_gga_", ""), mu=self.mu))
                elif name.startswith("sr_mgga"):
                    objs.append(RLibXCSRMGGA(name.replace("sr_mgga_", ""), mu=self.mu))
                elif name.startswith("lda"):
                    objs.append(RLibXCLDA(name.replace("lda_", "")))
                elif name.startswith("gga"):
                    objs.append(RLibXCGGA(name.replace("gga_", "")))
                elif name.startswith("mgga"):
                    objs.append(RLibXCMGGA(name.replace("mgga_", "")))
                elif name.startswith("hyb_gga"):
                    objs.append(RLibXCHybridGGA(name.replace("hyb_gga_", "")))
                elif name.startswith("hyb_mgga"):
                    objs.append(RLibXCHybridMGGA(name.replace("hyb_mgga_", "")))
                else:
                    raise RuntimeError("Unknown xc name: {}".format(name))
            else:
                # TODO: short-range functional for spin=1.
                # if name.startswith("sr_lda"):
                #     objs.append(RLibXCSRLDA(name.replace("sr_lda_", ""), mu=self.mu))
                # elif name.startswith("sr_gga"):
                #     objs.append(RLibXCSRGGA(name.replace("sr_gga_", ""), mu=self.mu))
                # elif name.startswith("sr_mgga"):
                #     objs.append(RLibXCSRMGGA(name.replace("sr_mgga_", ""), mu=self.mu))
                if name.startswith("lda"):
                    objs.append(ULibXCLDA(name.replace("lda_", "")))
                elif name.startswith("gga"):
                    objs.append(ULibXCGGA(name.replace("gga_", "")))
                elif name.startswith("mgga"):
                    objs.append(ULibXCMGGA(name.replace("mgga_", "")))
                elif name.startswith("hyb_gga"):
                    objs.append(ULibXCHybridGGA(name.replace("hyb_gga_", "")))
                elif name.startswith("hyb_mgga"):
                    objs.append(ULibXCHybridMGGA(name.replace("hyb_mgga_", "")))
                else:
                    raise RuntimeError("Unknown xc name: {}".format(name))

        return objs

    def get_xc_wrappers(self):
        """
        Get exchange-correlation wrapper used in Horton.

        Returns
        -------
        List:
            A list of xc wrapper defined in Horton.

        """
        objs = []
        for i, func_name in enumerate(self.func_names):
            if "sr" in func_name:
                if self.spin == 0:
                    objs.append(RLibXCSRWrapper(func_name))
                else:
                    raise NotImplementedError("Unrestricted short-range xc functional not found!")
            else:
                if self.spin == 0:
                    objs.append(RLibXCWrapper(func_name))
                else:
                    objs.append(ULibXCWrapper(func_name))
        return objs


def compute_lda_hardness_num(
    xc_wrapper,
    grid,
    rho_gs,
    rho_list,
    eps,
):
    """
    Compute LDA functional contribution to hardness numerically.

    Finite-difference method is employed here.

    Parameters
    ----------
    xc_wrapper: LibXCWrapper
        A LibXCWrapper for xc functional.
    grid: MoleculeGrid
        The molecular grid for integral.
    rho_gs: np.ndarray, shape=(N, )
        The ground-state density on a grid with `N` points.
    rho_list: np.ndarray, shape=(M, N)
        `M` density function on a grid with `N` points.
    eps: float
        Step for fintie difference.

    Returns
    -------
    hardness_xc: np.ndarray, shape=(M, M)
        The xc contribution to hardness.

    """
    nop = rho_list.shape[0]
    hardness_xc = np.zeros((nop, nop))

    for iop0 in range(nop):
        rhop = rho_gs + eps * rho_list[iop0]
        rhom = rho_gs - eps * rho_list[iop0]

        vrhop = np.zeros(grid.size)
        vrhom = np.zeros(grid.size)
        xc_wrapper.compute_lda_vxc(rhop, vrhop)
        xc_wrapper.compute_lda_vxc(rhom, vrhom)

        dvrho = (vrhop - vrhom) / (2 * eps)

        for iop1 in range(nop):
            hardness_xc[iop0, iop1] += grid.integrate(dvrho, rho_list[iop1])
    return hardness_xc


def compute_lda_hardness(xc_wrapper, grid, rho_gs, rho_list):
    """
    Compute LDA functional contribution to hardness analytically.

    The second derivative of xc energy w.r.t. density, i.e., :math:`f_{xc}` is used instead of
    numerically computing from xc potential.

    Parameters
    ----------
    xc_wrapper: LibXCWrapper
        The wrapper for xc functional.
    grid: MoleculeGrid
        Molecular grid.
    rho_gs: np.ndarray, shape=(N, )
        The ground-state density on a grid with `N` points.
    rho_list: np.ndarray, shape=(M, N)
        `M` density function on a grid with `N` points.

    Returns
    -------
    hardness_xc: np.ndarray, shape=(M, M)
        The xc contribution to hardness.

    """
    nop = rho_list.shape[0]
    hardness_xc = np.zeros((nop, nop))

    frr = np.zeros(grid.size)
    xc_wrapper.compute_lda_fxc(rho_gs, frr)

    for iop0 in range(nop):
        for iop1 in range(nop):
            hardness_xc[iop0, iop1] += grid.integrate(frr, rho_list[iop0] * rho_list[iop1])
    return hardness_xc


def compute_gga_hardness_num(
    xc_wrapper,
    grid,
    rho_gs,
    grad_gs,
    rho_list,
    grad_list,
    eps,
):
    """
    Compute GGA contribution to hardness numerically.

    Parameters
    ----------
    xc_wrapper
    grid
    rho_gs
    grad_gs
    rho_list
    grad_list
    eps
    xc_wrapper: LibXCWrapper
        A LibXCWrapper for xc functional.
    grid: MoleculeGrid
        The molecular grid for integral.
    rho_gs: np.ndarray, shape=(N, )
        The ground-state density on a grid with `N` points.
    grad_gs: np.ndarray, shape=(N, 3)
        The gradient of ground-state density.
    rho_list: np.ndarray, shape=(M, N)
        `M` density function on a grid with `N` points.
    grad_list: np.ndarray, shape=(M, N, 3)
        The gradient of `M` density function on a grid with `N` points.
    eps: float
        Step for fintie difference.

    Returns
    -------
    hardness_xc: np.ndarray, shape=(M, M)
        The xc contribution to hardness.

    """
    nop = rho_list.shape[0]
    hardness_xc = np.zeros((nop, nop))

    for iop0 in range(nop):
        rhop = rho_gs + eps * rho_list[iop0]
        rhom = rho_gs - eps * rho_list[iop0]

        vrhop = np.zeros(grid.size)
        vrhom = np.zeros(grid.size)
        vsigmap = np.zeros(grid.size)
        vsigmam = np.zeros(grid.size)

        gradp = grad_gs + eps * grad_list[iop0]
        gradm = grad_gs - eps * grad_list[iop0]
        sigmap = (gradp * gradp).sum(axis=1)
        sigmam = (gradm * gradm).sum(axis=1)
        xc_wrapper.compute_gga_vxc(rhop, sigmap, vrhop, vsigmap)
        xc_wrapper.compute_gga_vxc(rhom, sigmam, vrhom, vsigmam)
        gpotp = 2 * (gradp * vsigmap.reshape(-1, 1))
        gpotm = 2 * (gradm * vsigmam.reshape(-1, 1))

        dvrho = (vrhop - vrhom) / (2 * eps)
        dgpot = (gpotp - gpotm) / (2 * eps)

        for iop1 in range(nop):
            tmp = grid.integrate(dvrho, rho_list[iop1])
            tmp += grid.integrate(dgpot[:, 0], grad_list[iop1][:, 0])
            tmp += grid.integrate(dgpot[:, 1], grad_list[iop1][:, 1])
            tmp += grid.integrate(dgpot[:, 2], grad_list[iop1][:, 2])
            hardness_xc[iop0, iop1] += tmp
    return hardness_xc


def compute_gga_hardness(xc_wrapper, grid, rho_gs, grad_gs, rho_list, grad_list):
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    for alise in ["lda", "pbe", "srlda", "srpbe", "pbe0_13", "tpss", "lrcpbe"]:
        helper = XCHelper(alise, mu=0.6)
        print(helper.get_xc_wrappers())
        print(helper.get_xc_terms())

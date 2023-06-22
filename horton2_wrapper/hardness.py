from __future__ import division, print_function

import numpy as np
from horton2_wrapper.grid import solve_atomic_poisson_beck
from horton2_wrapper.xc import compute_lda_hardness_num, compute_gga_hardness_num
from horton2_wrapper.utils import contract

__all__ = [
    "Hardness",
    "compute_hartree_hardness",
    "compute_hartree_hardness_num",
]


class Hardness(object):
    """Hardness in ACKS2"""

    def __init__(self, scf, F_ao, eps=1e-7):
        """
        Atomic hardness defined in ACKS2.

        Parameters
        ----------
        F_ao: np.ndarray, shape=(M, N, N)
            Density matrix for `M` operator.
        eps: float, default=1e-7
            Step for fintie difference.

        """
        self.scf = scf
        self.grid = self.scf.grid
        self.dm_full = self.scf.compute_density_matrix(spin="ab")
        self.xc_helper = self.scf.xc_helper
        self.F_ao = F_ao
        self.eps = eps
        self.xc_wrappers = self.xc_helper.get_xc_wrappers()
        self.cache = {}

    @property
    def eta(self):
        """The total hardness."""
        return self.eta_xc + self.eta_h

    @property
    def eta_xc(self):
        """The xc contribution to hardness."""
        key = "hardness_xc"
        value = self.cache.get(key, None)
        if value is None:
            value = self._compute_xc_hardness_num()
            self.cache[key] = value
        return value

    @property
    def eta_h(self):
        """The Hartree hardness."""
        key = "hardness_hartree"
        value = self.cache.get(key, None)
        if value is None:
            value = compute_hartree_hardness_num(self.grid, self.rhos)
            self.cache[key] = value
        return value

    @property
    def rhos(self):
        """The density corresponding to density matrix F."""
        key = "rho_list"
        value = self.cache.get(key, None)
        if value is None:
            value = self.scf.compute_grid_density_dm(self.F_ao)
            self.cache[key] = value
        return value

    @property
    def grads(self):
        """The gradient of density corresponding to density matrix F."""
        key = "grad_list"
        value = self.cache.get(key, None)
        if value is None:
            value = self.scf.compute_grid_gradient_dm(self.F_ao)
            self.cache[key] = value
        return value

    @property
    def lapls(self):
        """The gradient of density corresponding to density matrix F."""
        key = "lapl_list"
        value = self.cache.get(key, None)
        if value is None:
            self._compute_mggas()
            value = self.cache[key]
        return value

    @property
    def taus(self):
        """The gradient of density corresponding to density matrix F."""
        key = "tau_list"
        value = self.cache.get(key, None)
        if value is None:
            self._compute_mggas()
            value = self.cache[key]
        return value

    def _compute_ggas(self):
        """The Laplace of density determined by density matrix F."""
        key = "gga_list"
        if key in self.cache:
            return

        value = self.scf.compute_grid_gga_dm(self.F_ao)
        self.cache["rho_list"] = value[:, :, 0]
        self.cache["grad_list"] = value[:, :, 1:4]
        self.cache[key] = True

    def _compute_mggas(self):
        """The Laplace of density determined by density matrix F."""
        key = "mgga_list"
        if key in self.cache:
            return

        value = self.scf.compute_grid_mgga_dm(self.F_ao)
        self.cache["rho_list"] = value[:, :, 0]
        self.cache["grad_list"] = value[:, :, 1:4]
        self.cache["lapl_list"] = value[:, :, 4]
        self.cache["tau_list"] = value[:, :, -1]
        self.cache[key] = True

    @property
    def rho_gs(self):
        """The ground-state density."""
        key = "rho_gs"
        value = self.cache.get(key, None)
        if value is None:
            value = self.scf.compute_density()
            self.cache[key] = value
        return value

    @property
    def grad_gs(self):
        """The gradient of ground-state density."""
        key = "grad_gs"
        value = self.cache.get(key, None)
        if value is None:
            value = self.scf.compute_gradient()
        return value

    @property
    def mgga_gs(self):
        """The laplace of ground-state density."""
        key = "mgga_gs"
        value = self.cache.get(key, None)
        if value is None:
            value = self.scf.compute_mgga()
        return value

    def _compute_xc_hardness_num(self):
        """
        Compute exchange-correlation hardness numerically

        Returns
        -------

        """
        eta_xc = 0.0
        for xc_wrapper, xc_name in zip(self.xc_wrappers, self.xc_helper.func_names):
            # actual computation with finite diffs, Eq. (28)
            if xc_wrapper.family in [1]:
                eta = compute_lda_hardness_num(
                    xc_wrapper, self.grid, self.rho_gs, self.rhos, self.eps
                )
            elif xc_wrapper.family in [2]:
                self._compute_ggas()
                eta = compute_gga_hardness_num(
                    xc_wrapper,
                    self.grid,
                    self.rho_gs,
                    self.grad_gs,
                    self.rhos,
                    self.grads,
                    self.eps,
                )
            elif xc_wrapper.family in [32]:
                raise NotImplementedError
            else:
                raise RuntimeError("No hardness for xc functional: {}".format(xc_name))

            eta = 0.5 * (eta + eta.T)
            self.cache["eta_{}".format(xc_name)] = eta
            eta_xc += eta
        return eta_xc

    def save(self, filename, compressed=True):
        """Save several arrays into a single file in compressed ``.npz`` format.

        Parameters
        ----------
        filename: str
            File name of ``.npz`` file.
        compressed: bool, default=True
            Check whether file is compressed.


        """
        data_to_save = {}
        data_to_save.update(self.cache)
        # data_to_save.update(self.chi_ks.cache)
        for key, value in self.cache.items():
            if key.startswith("eta") or key.startswith("hardness"):
                data_to_save[key] = value

        if compressed:
            np.savez_compressed(filename, **data_to_save)
        else:
            np.savez(filename, **data_to_save)


def compute_hartree_hardness_num(grid, rho_list, **kwargs):
    r"""
    Compute Hartree hardness numerically.

    The definition of Hartree hardness is given by:

    .. math:: \eta_{kl} = \iint dr dr' f_k(r) \frac{1}{|r-r'|} f_l(r')

    The idea of the numerical way is to get potential :math:`\phi(r)` by solving atomic
    poisson equation:

    .. math:: \phi(r) = \int \frac{f_k(r')}{|r-r'|} dr'


    Parameters
    ----------
    grid: MoleculeGrid
        A MoleculeGrid object. See `horton2_wrapper.wrapper.grid.MoleculeGrid`.
    rho_list: np.ndarray, shape=(M, N)
        The density basis function for `M` operator on the grid with a `N` points.
    kwargs: dict, optional
        The extra arguments for solving poisson equation.
        See `horton2_wrapper.wrapper.grid.solve_atomic_poisson_beck`
    Returns
    -------
    np.ndarray, shape=(M, M)
        The Hartree hardness.

    """
    nop = rho_list.shape[0]

    pot_list = []
    for i in range(nop):
        pots = solve_atomic_poisson_beck(grid, rho_list[i], len(grid.numbers), **kwargs)
        pot_list.append(pots)

    hardness = np.zeros((nop, nop))
    for i in range(nop):
        for j in range(i, nop):
            hardness[i, j] = grid.integrate(rho_list[i], pot_list[j])
            hardness[j, i] = hardness[i, j]
    # hardness[:] = 0.5 * (hardness + hardness.T)
    return hardness


def compute_hartree_hardness(F_ao, eri_ao):
    r"""Compute Hartree hardness analytically.

    The definition of Hartree hardness is given by:

    .. math:: \eta_{kl} = \iint dr dr' f_k(r) \frac{1}{|r-r'|} f_l(r')

    The basic idea is to expand density basis function on AO or MO basis, i.e.,

    .. math:: f_l(r) = \sum_{ij} \delta D_{ij}^l \phi_i^*(r) \phi_j(r)

    Then, the Hartree hardness :math:`\eta_{kl}` can be defined by

    .. math::

        \eta^H_{lk}
        = \iint \frac{f_l(r) f_k(r')}{||r-r'} dr dr'
        = \sum_{ij}\delta D_{ij}^l \delta D_{ij}^k
          \iint \frac{\phi_i^*(r) \phi_j(r) \phi^*_m(r')\phi_n(r')}{|r-r'|} drdr'
        = \sum_{ij} \delta D_{ij}^l D_{mn}^k \langle im|jn \rangle

    Returns
    -------
    """
    return contract("mij,ikjl,nkl->mn", F_ao, eri_ao, F_ao)

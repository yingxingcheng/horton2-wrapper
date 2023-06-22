from __future__ import division, print_function

import numpy as np
from horton import (
    RTwoIndexTerm,
    RDirectTerm,
    RGridGroup,
    RExchangeTerm,
    REffHam,
    IOData,
    guess_core_hamiltonian,
    PlainSCFSolver,
    CDIISSCFSolver,
    AufbauOccModel,
    transform_integrals,
)
from horton2_wrapper.molecule import Molecule
from horton2_wrapper.grid import MolecularGrid
from horton2_wrapper.xc import XCHelper

__all__ = ["SCF", "HFsrDFT", "DFT", "HF", "get_eri_mo_from_scf"]


def get_eri_mo_from_scf(scf, notation="physicist", use_long_range=False):
    """ERI on MO basis.

    Parameters
    ----------
    scf: SCF
        The SCF object.
    notation: {"physicist", "chemist"}, default="physicist"
        The notation for ERI.
    use_long_range: bool, default=False
        Whether a long-range ERI is used.

    Returns
    -------
    eri_mo: np.ndarray, shape=(N, N, N, N)
        The ERI on MO basis.

    """
    if use_long_range:
        eri_mo = scf.compute_eri_erf_mo(scf.mu)
    else:
        eri_mo = scf.eri_mo

    if notation == "chemist":
        eri_mo = eri_mo.swapaxes(1, 2)
    elif notation == "physicist":
        eri_mo = eri_mo
    else:
        raise NotImplementedError
    return eri_mo


class SCF(object):
    """
    The base module for SCF calculations.
    """

    def __init__(self, mol, xc="unknown", grid=None, mu=0.4):
        """
        The initialize function.

        Parameters
        ----------
        mol: Molecule
            See `horton2_wrapper.scf.molecule`
        xc: {'pbe', 'lda', 'b3lyp', 'tpss', 'm05'}, default='pbe'
            The type of exchange-correlation functional.
        grid: MoleculeGrid
            A molecular grid.
        mu: float, default=0.4
            A parameter for short-range (sr) DFT.
        """
        self.mol = mol
        self.xc = xc.lower()
        self.grid = grid or MolecularGrid.from_molecule(mol)
        self.mu = mu

        self.is_dft, self.is_hf, self.is_srdft = False, False, False
        if self.xc == "unknown":
            self.xc_helper = None
            self.xc_funcs = None
        else:
            if self.xc == "hf":
                self.xc_funcs = []
                self.is_hf = True
            else:
                self.xc_helper = XCHelper(xc, mu=mu)
                self.xc_funcs = self.xc_helper.get_xc_terms()
                self.is_srdft = self.xc_helper.has_srxc()
                self.is_dft = not self.is_srdft

        self.cache = {}

    @property
    def is_done(self):
        """Whether a SCF job has been finished."""
        return self.cache.get("is_done", False)

    @property
    def nocc(self):
        """Number of occupied orbitals."""
        return int(sum(self.mol.pseudo_numbers) // 2)

    @property
    def nvir(self):
        """The number of virtual orbitals."""
        return self.norb - self.nocc

    @property
    def norb(self):
        """The number of orbitals."""
        return self.Ca.shape[1]

    @property
    def ham(self):
        """Hamilton of SCF."""
        if "ham" not in self.cache:
            self.cache["ham"] = self._get_hamilton()
        return self.cache["ham"]

    @property
    def exp_alpha(self):
        """Coefficients of alpha orbitals."""
        return self.cache.get("exp_alpha", None)

    @property
    def exp_beta(self):
        """Coefficients of alpha orbitals."""
        return self.cache.get("exp_beta", self.exp_alpha)

    @property
    def energy(self):
        """The total SCF energy."""
        return self.cache.get("energy", None)

    @property
    def homo_index(self):
        """Index of alpha and beta HOMO orbital."""
        # HORTON indexes the orbitals from 0, so 1 is added to get the intuitive index
        return self.exp_alpha.get_homo_index() + 1, self.exp_beta.get_homo_index() + 1

    @property
    def lumo_index(self):
        """Index of alpha and beta LUMO orbital."""
        # HORTON indexes the orbitals from 0, so 1 is added to get the intuitive index
        return self.exp_alpha.get_lumo_index() + 1, self.exp_beta.get_lumo_index() + 1

    @property
    def homo_energy(self):
        """Energy of alpha and beta HOMO orbital."""
        return self.exp_alpha.homo_energy, self.exp_beta.homo_energy

    @property
    def lumo_energy(self):
        """Energy of alpha and beta LUMO orbital."""
        return self.exp_alpha.lumo_energy, self.exp_beta.lumo_energy

    @property
    def orbital_occupation(self):
        """Orbital occupation of alpha and beta electrons."""
        return self.exp_alpha.occupations, self.exp_beta.occupations

    @property
    def orbital_energy(self):
        """Orbital energy of alpha and beta electrons."""
        return self.exp_alpha.energies, self.exp_beta.energies

    @property
    def orbital_coefficient(self):
        """Orbital coefficient of alpha and beta electrons.

        The alpha and beta orbital coefficients are each storied in a 2d-array in which
        the columns represent the basis coefficients of each molecular orbital.
        """
        return self.exp_alpha.coeffs, self.exp_beta.coeffs

    @property
    def Ca(self):
        return self.exp_alpha.coeffs

    @property
    def Cb(self):
        return self.exp_beta.coeffs

    @property
    def epsilon_a(self):
        """Orbital energy of alpha electrons."""
        return self.exp_alpha.energies

    @property
    def epsilon_b(self):
        """Orbital energy of beta electrons."""
        return self.exp_beta.energies

    @property
    def eri(self):
        return self.mol.eri._array

    @property
    def eri_mo(self):
        """The electron-repulsion integrals on MO basis using physicist's notation."""
        one = self.mol.lf.create_two_index(self.mol.obasis.nbasis)
        (_,), (eri_mo,) = transform_integrals(one, self.mol.eri, "tensordot", self.exp_alpha)
        eri_mo = np.asarray(eri_mo._array)
        return eri_mo

    def compute_orbital_overlap(self):
        """Return the overlap matrix of molecular orbitals."""
        # compute overlap matrix
        return getattr(self.mol.olp, "_array")

    def compute_density_matrix(self, spin="ab"):
        """
        Return the density matrix array for the specified spin orbitals.
        """
        # get density matrix corresponding to the specified spin
        dm = self._get_density_matrix(spin)
        return getattr(dm, "_array")

    def _get_density_matrix(self, spin):
        """
        Return HORTON density matrix object corresponding to the specified spin.

        Parameters
        ----------
        spin : str
           The type of occupied spin orbitals. By default, the alpha and beta electrons (i.e.
           alpha and beta occupied spin orbitals) are used for computing the electron density.

           - "a" or "alpha": consider alpha electrons
           - "b" or "beta": consider beta electrons
           - "ab": consider alpha and beta electrons
        """
        # check orbital spin
        if spin not in ["a", "b", "alpha", "beta", "ab"]:
            raise ValueError("Argument spin={0} is not recognized!".format(spin))
        # compute density matrix
        if spin == "ab":
            # get density matrix of alpha & beta electrons
            dm = self.iodata.get_dm_full()
        else:
            # get orbital expression of specified spin
            spin_type = {"a": "alpha", "alpha": "alpha", "b": "beta", "beta": "beta"}
            exp = getattr(self, "exp_" + spin_type[spin])
            # get density matrix of specified spin
            dm = exp.to_dm()
        return dm

    def compute_grid_density_fock(self, pot, points=None, weights=None):
        """
        Compute a Fock operator from a density potential.

        **Warning:** the results are not added to the Fock operator!

        Parameters
        ----------
        points : np.ndarray, shape=(npoint, 3), dtype=float
            Cartesian grid points.
        weights : np.ndarray, shape=(npoint,), dtype=float
            Integration weights.
        pots : np.ndarray, shape=(npoint, 3), dtype=float
            Derivative of the energy toward the density gradient components at all grid
            points.

        """
        if points is None:
            points = self.grid.points
            weights = self.grid.weights

        fock = self.mol.lf.create_two_index()
        self.mol.obasis.compute_grid_density_fock(points, weights, pot, fock)
        return fock._array

    def compute_grid_gradient_fock(self, pot, points=None, weights=None):
        """
        Compute a Fock operator from a density gradient potential.

        **Warning:** the results are not added to the Fock operator!

        Parameters
        ----------
        points : np.ndarray, shape=(npoint, 3), dtype=float
            Cartesian grid points.
        weights : np.ndarray, shape=(npoint,), dtype=float
            Integration weights.
        pots : np.ndarray, shape=(npoint, 3), dtype=float
            Derivative of the energy toward the density gradient components at all grid
            points.

        """
        if points is None:
            points = self.grid.points
            weights = self.grid.weights

        fock = self.mol.lf.create_two_index()
        self.mol.obasis.compute_grid_gradient_fock(points, weights, pot, fock)
        return fock._array

    # TODO: compute_grid_other_fock()

    def compute_grid_density_dm(self, dm, points=None):
        """Compute density on a grid from density matrix.

        Parameters
        ----------
        dm: np.ndarray, shape=(M, M), dtype=float
            The density matrix.
        points: np.ndarray, shape=(N, ), dtype=float
            The grid points.

        Returns
        -------
        output: np.ndarray, shape=(N, )
            The density on the `N`-point grid.

        """
        if points is None:
            points = self.grid.points
        npoints = points.shape[0]

        if dm.ndim == 2:
            output = np.zeros((npoints,))
            horton_dm = self.mol.lf.create_two_index()
            horton_dm._array = dm
            self.mol.obasis.compute_grid_density_dm(horton_dm, points, output)
            outputs = output
        elif dm.ndim == 3:
            outputs = []
            ndm = dm.shape[0]
            for i in range(ndm):
                output = np.zeros((npoints,))
                horton_dm = self.mol.lf.create_two_index()
                # fixme: this is a bug in horton, the function cannot take subarray as input.
                tmp = np.zeros(dm[i].shape)
                tmp[:] = dm[i, :, :]
                horton_dm._array = tmp
                self.mol.obasis.compute_grid_density_dm(horton_dm, points, output)
                outputs.append(output)
        else:
            raise RuntimeError("The ndim of dm must be 2 or 3, but it is {}".format(dm.ndim))
        return np.asarray(outputs)

    def compute_grid_gradient_dm(self, dm, points=None):
        """Compute density gradient on a grid from density matrix.

        Parameters
        ----------
        dm: np.ndarray, shape=(M, M), dtype=float
            The density matrix.
        points: np.ndarray, shape=(N, ), dtype=float
            The grid points.

        Returns
        -------
        output: np.ndarray, shape=(N, 3)
            The density gradient on the `N`-point grid.

        """
        if points is None:
            points = self.grid.points

        if dm.ndim == 2:
            output = np.zeros((points.shape[0], 3))
            horton_dm = self.mol.lf.create_two_index()
            horton_dm._array = dm
            self.mol.obasis.compute_grid_gradient_dm(horton_dm, points, output)
            outputs = output
        else:
            outputs = []
            ndm = dm.shape[0]
            for i in range(ndm):
                output = np.zeros((points.shape[0], 3))
                horton_dm = self.mol.lf.create_two_index()
                # fixme: this is a bug in horton, the function cannot take subarray as input.
                tmp = np.zeros(dm[i].shape)
                tmp[:] = dm[i, :, :]
                horton_dm._array = tmp
                self.mol.obasis.compute_grid_gradient_dm(horton_dm, points, output)
                outputs.append(output)
        return np.asarray(outputs)

    def compute_grid_gga_dm(self, dm, points=None):
        """Compute GGA qualities on a grid from density matrix.

        Parameters
        ----------
        dm: np.ndarray, shape=(M, M), dtype=float
            The density matrix.
        points: np.ndarray, shape=(N, ), dtype=float
            The grid points.

        Returns
        -------
        output: np.ndarray, shape=(N, 3)
            The density gradient on the `N`-point grid.

        """
        if points is None:
            points = self.grid.points

        if dm.ndim == 2:
            output = np.zeros((points.shape[0], 4))
            horton_dm = self.mol.lf.create_two_index()
            horton_dm._array = dm
            self.mol.obasis.compute_grid_gga_dm(horton_dm, points, output)
            outputs = output
        else:
            outputs = []
            ndm = dm.shape[0]
            for i in range(ndm):
                output = np.zeros((points.shape[0], 4))
                horton_dm = self.mol.lf.create_two_index()
                # fixme: this is a bug in horton, the function cannot take subarray as input.
                tmp = np.zeros(dm[i].shape)
                tmp[:] = dm[i, :, :]
                horton_dm._array = tmp
                self.mol.obasis.compute_grid_gga_dm(horton_dm, points, output)
                outputs.append(output)
        return np.asarray(outputs)

    def compute_grid_kinetic_dm(self, dm, points=None):
        """Compute density on a grid from density matrix.

        Parameters
        ----------
        dm: np.ndarray, shape=(M, M), dtype=float
            The density matrix.
        points: np.ndarray, shape=(N, ), dtype=float
            The grid points.

        Returns
        -------
        output: np.ndarray, shape=(N, )
            The density on the `N`-point grid.

        """
        if points is None:
            points = self.grid.points
        npoints = points.shape[0]

        if dm.ndim == 2:
            output = np.zeros((npoints,))
            horton_dm = self.mol.lf.create_two_index()
            horton_dm._array = dm
            self.mol.obasis.compute_grid_kinetic_dm(horton_dm, points, output)
            outputs = output
        elif dm.ndim == 3:
            outputs = []
            ndm = dm.shape[0]
            for i in range(ndm):
                output = np.zeros((npoints,))
                horton_dm = self.mol.lf.create_two_index()
                # fixme: this is a bug in horton, the function cannot take subarray as input.
                tmp = np.zeros(dm[i].shape)
                tmp[:] = dm[i, :, :]
                horton_dm._array = tmp
                self.mol.obasis.compute_grid_kinetic_dm(horton_dm, points, output)
                outputs.append(output)
        else:
            raise RuntimeError("The ndim of dm must be 2 or 3, but it is {}".format(dm.ndim))
        return np.asarray(outputs)

    def compute_grid_mgga_dm(self, dm, points=None):
        """Compute mgga qualities on a grid from density matrix.

        Parameters
        ----------
        dm: np.ndarray, shape=(M, M), dtype=float
            The density matrix.
        points: np.ndarray, shape=(N, ), dtype=float
            The grid points.

        Returns
        -------
        output: np.ndarray, shape=(N, 3)
            The density gradient on the `N`-point grid.

        """
        if points is None:
            points = self.grid.points

        if dm.ndim == 2:
            output = np.zeros((points.shape[0], 6))
            horton_dm = self.mol.lf.create_two_index()
            horton_dm._array = dm
            self.mol.obasis.compute_grid_mgga_dm(horton_dm, points, output)
            outputs = output
        else:
            outputs = []
            ndm = dm.shape[0]
            for i in range(ndm):
                output = np.zeros((points.shape[0], 6))
                horton_dm = self.mol.lf.create_two_index()
                # fixme: this is a bug in horton, the function cannot take subarray as input.
                tmp = np.zeros(dm[i].shape)
                tmp[:] = dm[i, :, :]
                horton_dm._array = tmp
                self.mol.obasis.compute_grid_mgga_dm(horton_dm, points, output)
                outputs.append(output)
        return np.asarray(outputs)

    # TODO compute_grid_other_dm()

    def compute_molecular_orbital(self, points=None, spin="a", index=None, output=None):
        """
        Return molecular orbitals evaluated on the given points for the spin orbitals.

        Parameters
        ----------
        points : ndarray
           The 2d-array containing the cartesian coordinates of points on which density is
           evaluated. It has a shape (n, 3) where n is the number of points.
        spin : str
           The type of occupied spin orbitals.

           - "a" or "alpha": consider alpha electrons
           - "b" or "beta": consider beta electrons

        index : sequence, default=None
           Sequence of integers representing the index of spin orbitals. Alpha and beta spin
           orbitals are each indexed from 1 to :attr:`nbasis`.
           If ``None``, all occupied spin orbitals are included.
        output : np.ndarray, default=None
           Array with shape (n, m) to store the output, where n in the number of points and m
           is the number of molecular orbitals. When ``None`` the array is allocated.
        """
        if points is None:
            points = self.grid.points
        # check points
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Argument points should be a 2d-array with 3 columns.")
        if not np.issubdtype(points.dtype, np.float64):
            raise ValueError("Argument points should be a 2d-array of floats!")

        # assign orbital index (HORTON index the orbitals from 0)
        if index is None:
            # include all occupied orbitals of specified spin
            spin_index = {"a": 0, "alpha": 0, "b": 1, "beta": 1}
            index = np.arange(self.homo_index[spin_index[spin]])
        else:
            # include specified set of orbitals
            index = np.copy(np.asarray(index)) - 1
            if index.ndim == 0:
                index = np.array([index])
            if np.any(index < 0):
                raise ValueError("Argument index={0} cannot be less than one!".format(index + 1))

        # allocate output array
        if output is None:
            output = np.zeros((points.shape[0], index.shape[0]), float)
        npoints, norbs = points.shape[0], index.shape[0]
        if output.shape != (npoints, norbs):
            raise ValueError("Argument output should be a {0} array.".format((npoints, norbs)))

        # get orbital expression of specified spin
        spin_type = {"a": "alpha", "alpha": "alpha", "b": "beta", "beta": "beta"}
        exp = getattr(self, "_exp_" + spin_type[spin])
        # compute mo expression
        self.mol.obasis.compute_grid_orbitals_exp(exp, points, index, output=output)
        return output

    def compute_density(self, points=None, spin="ab", index=None, output=None):
        r"""
        Return electron density evaluated on the given points for the spin orbitals.

        Parameters
        ----------
        points : ndarray
           The 2d-array containing the cartesian coordinates of points on which density is
           evaluated. It has a shape (n, 3) where n is the number of points.
        spin : str
           The type of occupied spin orbitals. By default, the alpha and beta electrons (i.e.
           alpha and beta occupied spin orbitals) are used for computing the electron density.

           - "a" or "alpha": consider alpha electrons
           - "b" or "beta": consider beta electrons
           - "ab": consider alpha and beta electrons

        index : sequence
           Sequence of integers representing the index of spin orbitals. Alpha and beta spin
           orbitals are each indexed from 1 to :attr:`nbasis`.
           If ``None``, all occupied spin orbitals are included.
        output : np.ndarray
           Array with shape (n,) to store the output, where n in the number of points.
           When ``None`` the array is allocated.
        """
        if points is None:
            points = self.grid.points
        # check points
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Argument points should be a 2d-array with 3 columns.")
        if not np.issubdtype(points.dtype, np.float64):
            raise ValueError("Argument points should be a 2d-array of floats!")

        # allocate output array
        if output is None:
            output = np.zeros((points.shape[0],), float)
        if output.shape != (points.shape[0],):
            raise ValueError("Argument output should be a {0} array.".format((points.shape[0],)))

        # compute density
        if index is None:
            # get density matrix corresponding to the specified spin
            dm = self._get_density_matrix(spin)
            # include all orbitals
            self.mol.obasis.compute_grid_density_dm(dm, points, output=output)
        else:
            # include subset of molecular orbitals
            if spin == "ab":
                # compute mo expression of alpha & beta orbitals
                mo_a = self.compute_molecular_orbital(points, "a", index)
                mo_b = self.compute_molecular_orbital(points, "b", index)
                # add density of alpha & beta molecular orbitals
                np.sum(mo_a**2, axis=1, out=output)
                output += np.sum(mo_b**2, axis=1)
            else:
                # compute mo expression of specified molecular orbitals
                mo = self.compute_molecular_orbital(points, spin, index)
                # add density of specified molecular orbitals
                np.sum(mo**2, axis=1, out=output)
        return output

    def compute_gradient(self, points=None, spin="ab", index=None, output=None):
        r"""
        Return gradient of electron density evaluated on the given points for the spin orbitals.

        Parameters
        ----------
        points : ndarray
           The 2d-array containing the cartesian coordinates of points on which density is
           evaluated. It has a shape (n, 3) where n is the number of points.
        spin : str
           The type of occupied spin orbitals. By default, the alpha and beta electrons (i.e.
           alpha and beta occupied spin orbitals) are used for computing the electron density.

           - "a" or "alpha": consider alpha electrons
           - "b" or "beta": consider beta electrons
           - "ab": consider alpha and beta electrons

        index : sequence
           Sequence of integers representing the index of spin orbitals. Alpha and beta spin
           orbitals are each indexed from 1 to :attr:`nbasis`.
           If ``None``, all occupied spin orbitals are included.
        output : np.ndarray
           Array with shape (n, 3) to store the output, where n in the number of points.
           When ``None`` the array is allocated.
        """
        if points is None:
            points = self.grid.points
        # check points
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Argument points should be a 2d-array with 3 columns.")
        if not np.issubdtype(points.dtype, np.float64):
            raise ValueError("Argument points should be a 2d-array of floats!")

        # allocate output array
        if output is None:
            output = np.zeros((points.shape[0], 3), float)
        if output.shape != (points.shape[0], 3):
            raise ValueError("Argument output should be a {0} array.".format((points.shape[0], 3)))

        # get density matrix corresponding to the specified spin
        dm = self._get_density_matrix(spin)
        # compute gradient
        if index is None:
            # include all orbitals
            self.mol.obasis.compute_grid_gradient_dm(dm, points, output=output)
        else:
            # include specified set of orbitals
            raise NotImplementedError()
        return output

    def compute_hessian(self, points=None, spin="ab", index=None, output=None):
        r"""
        Return hessian of electron density evaluated on the given points for the spin orbitals.

        Parameters
        ----------
        points : ndarray
           The 2d-array containing the cartesian coordinates of points on which density is
           evaluated. It has a shape (n, 3) where n is the number of points.
        spin : str
           The type of occupied spin orbitals. By default, the alpha and beta electrons (i.e.
           alpha and beta occupied spin orbitals) are used for computing the electron density.

           - "a" or "alpha": consider alpha electrons
           - "b" or "beta": consider beta electrons
           - "ab": consider alpha and beta electrons

        index : sequence
           Sequence of integers representing the index of spin orbitals. Alpha and beta spin
           orbitals are each indexed from 1 to :attr:`nbasis`.
           If ``None``, all occupied spin orbitals are included.
        output : np.ndarray
           Array with shape (n, 6) to store the output, where n in the number of points.
           When ``None`` the array is allocated.
        """
        if points is None:
            points = self.grid.points
        # check points
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Argument points should be a 2d-array with 3 columns.")
        if not np.issubdtype(points.dtype, np.float64):
            raise ValueError("Argument points should be a 2d-array of floats!")

        # allocate output array
        if output is None:
            output = np.zeros((points.shape[0], 6), float)
        if output.shape != (points.shape[0], 6):
            raise ValueError("Argument output should be a {0} array.".format((points.shape[0], 6)))

        # get density matrix corresponding to the specified spin
        dm = self._get_density_matrix(spin)
        # compute hessian
        if index is None:
            # include all orbitals
            self.mol.obasis.compute_grid_hessian_dm(dm, points, output=output)
        else:
            # include specified set of orbitals
            raise NotImplementedError()
        return output

    def compute_esp(self, points=None, spin="ab", index=None, output=None, charges=None):
        r"""
        Return the molecular electrostatic potential on the given points for the specified spin.

        The molecular electrostatic potential at point :math:`\mathbf{r}` is caused by the
        electron density :math:`\rho` of the specified spin orbitals and set of point charges
        :math:`\{q_A\}_{A=1}^{N_\text{atoms}}` placed at the position of the nuclei. i.e,

        .. math::
           V \left(\mathbf{r}\right) =
             \sum_{A=1}^{N_\text{atoms}} \frac{q_A}{\rvert \mathbf{R}_A - \mathbf{r} \lvert} -
             \int \frac{\rho \left(\mathbf{r}"\right)}{\rvert \mathbf{r}" - \mathbf{r} \lvert}
                  d\mathbf{r}"

        Parameters
        ----------
        points : ndarray
           The 2d-array containing the cartesian coordinates of points on which density is
           evaluated. It has a shape (n, 3) where n is the number of points.
        spin : str, default="ab"
           The type of occupied spin orbitals. By default, the alpha and beta electrons (i.e.
           alpha and beta occupied spin orbitals) are used for computing the electron density.

           - "a" or "alpha": consider alpha electrons
           - "b" or "beta": consider beta electrons
           - "ab": consider alpha and beta electrons

        index : sequence, default=None
           Sequence of integers representing the index of spin orbitals. Alpha and beta spin
           orbitals are each indexed from 1 to :attr:`nbasis`.
           If ``None``, all occupied spin orbitals are included.
        output : np.ndarray, default=None
           Array with shape (n,) to store the output, where n in the number of points.
           When ``None`` the array is allocated.
        charges : np.ndarray, default=None
           Array with shape (n,) representing the point charges at the position of the nuclei.
           When ``None``, the pseudo numbers are used.
        """
        if points is None:
            points = self.grid.points
        # check points
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Argument points should be a 2d-array with 3 columns.")
        if not np.issubdtype(points.dtype, np.float64):
            raise ValueError("Argument points should be a 2d-array of floats!")

        # allocate output array
        if output is None:
            output = np.zeros((points.shape[0],), np.float)
        if output.shape != (points.shape[0],):
            raise ValueError("Argument output should be a {0} array.".format((points.shape[0],)))

        # get density matrix corresponding to the specified spin
        dm = self._get_density_matrix(spin)
        # assign point charges
        if charges is None:
            charges = self.mol.pseudo_numbers
        elif not isinstance(charges, np.ndarray) or charges.shape != self.mol.numbers.shape:
            raise ValueError(
                "Argument charges should be a 1d-array "
                "with {0} shape.".format(self.mol.numbers.shape)
            )
        # compute esp
        if index is None:
            # include all orbitals
            self.mol.obasis.compute_grid_esp_dm(
                dm, self.mol.coordinates, charges, points, output=output
            )
        else:
            # include specified set of orbitals
            raise NotImplementedError()
        return output

    def compute_ked(self, points=None, spin="ab", index=None, output=None):
        r"""
        Return positive definite kinetic energy density on the given points for the specified spin.

        Positive definite kinetic energy density is defined as,

        .. math::
           \tau \left(\mathbf{r}\right) =
           \sum_i^N n_i \frac{1}{2} \rvert \nabla \phi_i \left(\mathbf{r}\right) \lvert^2

        Parameters
        ----------
        points : ndarray
           The 2d-array containing the cartesian coordinates of points on which density is
           evaluated. It has a shape (n, 3) where n is the number of points.
        spin : str
           The type of occupied spin orbitals. By default, the alpha and beta electrons (i.e.
           alpha and beta occupied spin orbitals) are used for computing the electron density.

           - "a" or "alpha": consider alpha electrons
           - "b" or "beta": consider beta electrons
           - "ab": consider alpha and beta electrons

        index : sequence
           Sequence of integers representing the index of spin orbitals. Alpha and beta spin
           orbitals are each indexed from 1 to :attr:`nbasis`.
           If ``None``, all occupied spin orbitals are included.
        output : np.ndarray
           Array with shape (n,) to store the output, where n in the number of points.
           When ``None`` the array is allocated.
        """
        if points is None:
            points = self.grid.points
        # check points
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Argument points should be a 2d-array with 3 columns.")
        if not np.issubdtype(points.dtype, np.float64):
            raise ValueError("Argument points should be a 2d-array of floats!")
        # allocate output array
        if output is None:
            output = np.zeros((points.shape[0],), float)
        if output.shape != (points.shape[0],):
            raise ValueError("Argument output should be a {0} array.".format((points.shape[0],)))
        # get density matrix corresponding to the specified spin
        dm = self._get_density_matrix(spin)
        # compute kinetic energy
        if index is None:
            # include all orbitals
            self.mol.obasis.compute_grid_kinetic_dm(dm, points, output=output)
        else:
            # include specified set of orbitals
            raise NotImplementedError()
        return output

    def compute_megga(self, points=None, spin="ab", index=None):
        """Return electron density, gradient, laplacian & kinetic energy density.

        Parameters
        ----------
        points : ndarray
           The 2d-array containing the cartesian coordinates of points on which density is
           evaluated. It has a shape (n, 3) where n is the number of points.
        spin : str
           The type of occupied spin orbitals. By default, the alpha and beta electrons (i.e.
           alpha and beta occupied spin orbitals) are used for computing the electron density.

           - "a" or "alpha": consider alpha electrons
           - "b" or "beta": consider beta electrons
           - "ab": consider alpha and beta electrons

        index : sequence
           Sequence of integers representing the index of spin orbitals. Alpha and beta spin
           orbitals are each indexed from 1 to :attr:`nbasis`.
           If ``None``, all occupied spin orbitals are included.
        """
        if points is None:
            points = self.grid.points
        # check points
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Argument points should be a 2d-array with 3 columns.")
        if not np.issubdtype(points.dtype, np.float64):
            raise ValueError("Argument points should be a 2d-array of floats!")

        # get density matrix corresponding to the specified spin
        dm = self._get_density_matrix(spin)

        # compute for the given set of orbitals
        if index is None:
            output = self.mol.obasis.compute_grid_mgga_dm(dm, points)
        else:
            raise NotImplementedError()
        return output[:, 0], output[:, 1:4], output[:, 4], output[:, 5]

    def compute_kin_fock(self):
        """Return the kinetic matrix of molecular orbitals."""
        return getattr(self.mol.kin, "_array")

    def compute_na_fock(self):
        """Return the kinetic matrix of molecular orbitals."""
        return getattr(self.mol.na, "_array")

    def compute_eri_ao(self):
        """Return the electron-repulsion matrix of molecular orbitals."""
        return getattr(self.mol.eri, "_array")

    def compute_nn_fock(self):
        """Return the nucnuc matrix of molecular orbitals."""
        return getattr(self.mol.nn, "_array")

    def compute_eri_erf_ao(self, mu):
        r"""Compute long-range ERI on AO basis.

        .. math:: g(r_{12}, \mu) = \iint \frac{\erf(\mu r_{12})}{|r_12|}

        Parameters
        ----------
        mu: float
            The range-separation parameter.
        """
        return getattr(self.mol.eri_erf(mu), "_array")

    def compute_eri_erf_mo(self, mu):
        """Long-range ERI defined by Erf on MO basis."""
        one = self.mol.lf.create_two_index(self.mol.obasis.nbasis)
        (_,), (eri_mo,) = transform_integrals(
            one, self.mol.eri_erf(mu), "tensordot", self.exp_alpha
        )
        eri_mo = np.asarray(eri_mo._array)
        # the default format in Horton is physical notation
        return eri_mo

    def run(self, scf_solver_type="cdiis", threshold=1e-6):
        """Main entrance."""
        if self.is_done:
            return self.energy

        # Create alpha orbitals
        exp_alpha = self.mol.lf.create_expansion()

        # Initial guess
        guess_core_hamiltonian(self.mol.olp, self.mol.kin, self.mol.na, exp_alpha)

        # Decide how to occupy the orbitals (5 alpha electrons)
        # nocc = int(sum(self.mol.pseudo_numbers) // 2)
        occ_model = AufbauOccModel(self.nocc)

        if scf_solver_type == "plain":
            scf_solver = PlainSCFSolver(threshold=threshold)
            scf_solver(self.ham, self.mol.lf, self.mol.olp, occ_model, exp_alpha)
            dm_alpha = exp_alpha.to_dm()
        elif scf_solver_type == "cdiis":
            # Converge WFN with CDIIS SCF
            # - Construct the initial density matrix (needed for CDIIS).
            occ_model.assign(exp_alpha)
            dm_alpha = exp_alpha.to_dm()
            # - SCF solver
            scf_solver = CDIISSCFSolver(threshold=threshold, maxiter=500)
            scf_solver(self.ham, self.mol.lf, self.mol.olp, occ_model, dm_alpha)

            # Derive orbitals (coeffs, energies and occupations) from the Fock and density
            # matrices. The energy is also computed to store it in the output file below.
            fock_alpha = self.mol.lf.create_two_index()
            self.ham.reset(dm_alpha)
            self.ham.compute_energy()
            self.ham.compute_fock(fock_alpha)
            exp_alpha.from_fock_and_dm(fock_alpha, dm_alpha, self.mol.olp)
        else:
            raise RuntimeError("Unknown scf solver type : {}".format(scf_solver_type))

        for i in range(len(exp_alpha.occupations)):
            v = exp_alpha.occupations[i]
            if np.isclose(v, 1.0, atol=1e-4, rtol=1e-4):
                exp_alpha.occupations[i] = 1.0
                continue
            if np.isclose(v, 0.0, atol=1e-4, rtol=1e-4):
                exp_alpha.occupations[i] = 0.0

        for k, v in self.ham.cache.iteritems():
            if "energy" in k:
                self.cache[k] = v

        self.cache["energy"] = self.ham.cache["energy"]
        self.cache["exp_alpha"] = exp_alpha
        self.cache["is_done"] = True
        return self.cache["energy"]

    def _get_hamilton(self):
        """Interface to construct hamilton of SCF."""
        external = {"nn": self.mol.nn}

        terms = [
            RTwoIndexTerm(self.mol.kin, "kin"),
            RDirectTerm(self.mol.eri, "hartree"),
            RTwoIndexTerm(self.mol.na, "ne"),
        ]

        if self.is_hf:
            terms.append(RExchangeTerm(self.mol.eri, "x_hf"))
        else:
            terms.append(RGridGroup(self.mol.obasis, self.grid, self.xc_funcs))

        if self.is_srdft:
            terms.append(RExchangeTerm(self.mol.eri_erf(mu=self.mu), "x_hf"))

        if self.is_dft:
            if hasattr(self.xc_funcs[0], "get_exx_fraction"):
                exx_fraction = self.xc_funcs[0].get_exx_fraction()
                if not np.isclose(exx_fraction, 0.0):
                    terms.append(RExchangeTerm(self.mol.eri, "x_hf", exx_fraction))

        ham = REffHam(terms, external)
        return ham

    def compute_exx(self):
        """The function to compute EXact-exchange (EXX) energy using DFT orbitals.

        Note: the EXX method mentioned here should be differentiated from EXX derived by OEP method.
        """
        terms = [
            RExchangeTerm(self.mol.eri, "x_hf"),
        ]
        ham = REffHam(terms)
        dm_alpha = self.cache.get("exp_alpha").to_dm()
        ham.reset(dm_alpha)
        ham.compute_energy()
        e_exx = ham.cache["energy"]
        # self.mol.energy_exx = e_exx
        return e_exx

    def reproduce_result(self):
        """Reproduce results from density matrix."""
        exp_alpha = self.cache.get("exp_alpha", None)
        dm_alpha = exp_alpha.to_dm()
        assert dm_alpha is not None
        fock_alpha = self.mol.lf.create_two_index()
        self.ham.reset(dm_alpha)
        self.ham.compute_energy()
        self.ham.compute_fock(fock_alpha)

        res = {}
        for k, v in self.ham.cache.iteritems():
            if "energy" in k:
                res[k] = v
                print("{:<25} : {:>20.6f}".format(k, v))
        return res

    # @property
    # def mol(self):
    #     """Generate IOData for post-SCF methods.
    #     Returns
    #     -------
    #     IOData
    #     """
    #     data = IOData()
    #     data.title = '{} calculations'.format(self.__class__.__name__)
    #     data.coordinates = self.coordinates
    #     data.numbers = self.numbers
    #     data.pseudo_numbers = self.pseudo_numbers
    #     data.exp_alpha = self.exp_alpha
    #     data.energy = self.energy
    #     data.lf = self.lf
    #     data.obasis = self.obasis
    #     data.grid = self.grid
    #     data.dm_alpha = self.dm_alpha

    #     return data

    @property
    def iodata(self):
        data = IOData()
        mol_keys = ["coordinates", "lf", "obasis", "pseudo_numbers", "numbers"]
        scf_keys = ["energy", "exp_alpha"]
        for k in mol_keys:
            setattr(data, k, getattr(self.mol, k))
        for k in scf_keys:
            setattr(data, k, getattr(self, k))
        return data

    def to_file(self, filename):
        """Write molecular info to a file."""
        self.iodata.to_file(filename)

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Construct object from a Gaussian checkpoint file."""
        return cls.from_iodata(data=IOData.from_file(filename), **kwargs)

    @classmethod
    def from_iodata(cls, data, xc="unknown", **kwargs):
        """
        Construct from IOData.

        If a GOBasis object is included in IOData, then we assume that the DFT calculation is
        finished, and read info from the IOData. Otherwise, a new DFT object is constructed with
        `is_done` being False.

        Parameters
        ----------
        data: IOData
           See `horton.IOData`.
        xc: {'pbe', 'lda', 'b3lyp', 'tpss', 'm05'}, default='pbe'
            The type of xc functional.

        Returns
        -------
        DFT

        """
        # check validity
        keys = ["energy", "exp_alpha", "lf", "obasis"]
        is_ok = True
        for k in keys:
            if not hasattr(data, k):
                is_ok = False
                break

        if is_ok:
            mol = Molecule.from_iodata(data, basis="unknown")
            mol._lf = getattr(data, "lf")
            mol._obasis = getattr(data, "obasis")
            # if params of grid are provided
            specification = kwargs.get("specification", "medium")
            k = kwargs.get("k", 3)
            rotate = kwargs.get("rotate", False)
            grid = MolecularGrid.from_molecule(mol, specification=specification, k=k, rotate=rotate)
            # if mu provided
            mu = kwargs.get("mu", 0.4)
            obj = cls(mol=mol, xc=xc, grid=grid, mu=mu)
            obj.cache["is_done"] = True
            obj.cache["energy"] = getattr(data, "energy")
            obj.cache["exp_alpha"] = getattr(data, "exp_alpha")
        else:
            raise RuntimeError("The data is not valid!")
        return obj

    @classmethod
    def HF(cls, mol):
        """Hartree-Fock method can be regarded as a special case of DFT."""
        return cls(mol=mol, xc="hf", grid=None)


HFsrDFT = DFT = SCF
HF = SCF.HF

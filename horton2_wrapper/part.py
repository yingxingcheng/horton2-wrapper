# -*- coding: utf-8 -*-
# ChemTools is a collection of interpretive chemical tools for
# analyzing outputs of the quantum chemistry calculations.
#
# Copyright (C) 2016-2019 The ChemTools Development Team
#
# This file is part of ChemTools.
#
# ChemTools is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# ChemTools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Wrapper of Part Module."""

import numpy as np
from horton import ProAtomDB, get_npure_cumul, fill_pure_polynomials
from horton.scripts.wpart import wpart_schemes
from horton2_wrapper.scf import SCF

__all__ = ["DensPart", "compute_molecular_multipole_moment", "iter_cartesian_powers"]


class DensPart(object):
    """Density partitioning class."""

    def __init__(self, scf, grid=None, scheme="b", spin="ab", **kwargs):
        """
        Density Partitioning object.

        Parameters
        ----------
        scf: SCF
            A `SCF` object, e.g., HF, DFT, or HFsrDFT. See `horton2_wrapper.wrapper.scf.SCF`.
        grid: MolecularGrid
            See `horton2_wrapper.wrapper.grid.MolecularGrid`.
        scheme: {"b", "mbis"}
            The alise of partitioning scheme, default is MBIS, i.e., "b" or "mbis".
        spin: {"ab", "a", "b"}
            The spin of the molecule.
        kwargs: dict, optional
            The extra arguments for partitioning.
        """
        self.scf = scf
        grid = grid or scf.grid
        density = scf.compute_density(grid.points, spin=spin)

        wpart = wpart_schemes[scheme]
        # make proatom database
        if scheme.lower() not in ["mbis", "b"]:
            if "proatomdb" not in kwargs.keys():
                proatomdb = ProAtomDB.from_refatoms(scf.mol.numbers)
            else:
                proatomdb = kwargs["proatomdb"]
            kwargs["proatomdb"] = proatomdb
        # partition
        self.part = wpart(
            scf.mol.coordinates, scf.mol.numbers, scf.mol.pseudo_numbers, grid, density, **kwargs
        )
        self.part.do_all()

        self.density = density
        self.coordinates = scf.mol.coordinates
        self.numbers = scf.mol.numbers
        self.pseudo_numbers = scf.mol.pseudo_numbers
        self.charges = self.part["charges"]

        self._at_weights = None
        self._at_grids = None
        self._at_grid_weights = None

    def get_at_weights(self, index):
        """A dict contains atom weights with atom index as keys."""
        return np.asarray(self.part.cache.load("at_weights", index))

    def get_at_grid(self, index):
        """Get atomic grid by specifying the atomic index."""
        return self.part.get_grid(index)

    def compute_aim_transition_multipole_moment(self, lmax):
        r"""Compute Fock matrix of potential function based on AIM function.

        .. math::

            \langle \phi_i | \omega_a(r) R_k(r) | \phi_j \rangle

        where :math:`\phi_i` is orbital `i`, :math:`\omega_a(r)` is weight function of
        atom `a` and :math:`R_k(r)` is real spherical harmonics.

        """
        # compute overlap operators
        olp = self.scf.mol.olp
        npure = get_npure_cumul(lmax)

        overlap_operators = {}
        for iatom in range(self.part.natom):
            # Prepare solid harmonics on grids.
            # grid = self.part.get_grid(iatom)
            grid = self.get_at_grid(iatom)
            if lmax > 0:
                work = np.zeros((grid.size, npure - 1), float)
                work[:, 0] = grid.points[:, 2] - self.coordinates[iatom, 2]
                work[:, 1] = grid.points[:, 0] - self.coordinates[iatom, 0]
                work[:, 2] = grid.points[:, 1] - self.coordinates[iatom, 1]
                if lmax > 1:
                    fill_pure_polynomials(work, lmax)
            else:
                work = None

            # at_weights = self.part.cache.load("at_weights", iatom)
            at_weights = self.get_at_weights(iatom)
            # Convert the weight functions to AIM overlap operators.
            for ipure in range(npure):
                if ipure > 0:
                    tmp = at_weights * work[:, ipure - 1]
                else:
                    tmp = at_weights
                # convert weight functions to matrix based on basis sets
                # op = self.scf.mol.lf.create_two_index()
                # self.scf.mol.obasis.compute_grid_density_fock(grid.points, grid.weights, tmp, op)
                op = self.scf.compute_grid_density_fock(tmp, grid.points, grid.weights)
                overlap_operators[(iatom, ipure)] = op

        # Correct the s-type overlap operators such that the sum is exactly equal to the total
        # overlap.
        calc_olp = 0.0
        for i in range(self.part.natom):
            calc_olp += overlap_operators[(i, 0)]
        error_olp = (calc_olp - olp._array) / self.part.natom
        for i in range(self.part.natom):
            overlap_operators[(i, 0)] - error_olp

        # sort the operators
        result = []
        # sort the response function basis
        for ipure in range(npure):
            for iatom in range(self.part.natom):
                result.append(overlap_operators[(iatom, ipure)])
        return np.asarray(result)

    def compute_atomic_transition_multipole_moment(self, lmax):
        r"""Compute Fock matrix of potential function based on AIM function.

        .. math::

            \langle \phi_i | \omega_a(r) R_k(r) | \phi_j \rangle

        where :math:`\phi_i` is orbital `i`, :math:`\omega_a(r)` is weight function of
        atom `a` and :math:`R_k(r)` is real spherical harmonics.

        """
        npure = get_npure_cumul(lmax)
        overlap_operators = {}
        for iatom in range(self.part.natom):
            # Prepare solid harmonics on grids.
            # grid = self.part.get_grid(iatom)
            grid = self.get_at_grid(iatom)
            if lmax > 0:
                work = np.zeros((grid.size, npure - 1), float)
                work[:, 0] = grid.points[:, 2]
                work[:, 1] = grid.points[:, 0]
                work[:, 2] = grid.points[:, 1]
                if lmax > 1:
                    fill_pure_polynomials(work, lmax)
            else:
                work = None

            # at_weights = self.part.cache.load("at_weights", iatom)
            at_weights = self.get_at_weights(iatom)
            # Convert the weight functions to AIM overlap operators.
            for ipure in range(npure):
                if ipure > 0:
                    tmp = at_weights * work[:, ipure - 1]
                else:
                    tmp = at_weights
                # convert weight functions to matrix based on basis sets
                # op = self.scf.mol.lf.create_two_index()
                # self.scf.mol.obasis.compute_grid_density_fock(grid.points, grid.weights, tmp, op)
                op = self.scf.compute_grid_density_fock(tmp, grid.points, grid.weights)
                overlap_operators[(iatom, ipure)] = op

        # sort the operators
        result = []
        # sort the response function basis
        for ipure in range(1, npure):
            for iatom in range(self.scf.mol.natom):
                result.append(overlap_operators[(iatom, ipure)])
        return np.asarray(result)

    @classmethod
    def from_file(cls, fname, scheme=None, grid=None, spin="ab", **kwargs):
        """Initialize class given a file."""
        scf = SCF.from_file(fname)
        return cls(scf, scheme=scheme, grid=grid, spin=spin, **kwargs)

    def condense_to_atoms(self, property):
        """Condense properties to atomic contribution.

        Parameters
        ----------
        property: np.ndarray, shape=(N, )
            Data on grid with `N` points.

        Returns
        -------
        condensed: np.ndarray, shape=(M, )
            Property for atom `M`.

        """
        condensed = np.zeros(self.part.natom)
        for index in range(self.part.natom):
            at_grid = self.part.get_grid(index)
            at_weight = self.part.cache.load("at_weights", index)
            wcor = self.part.get_wcor(index)
            local_prop = self.part.to_atomic_grid(index, property)
            condensed[index] = at_grid.integrate(at_weight, local_prop, wcor)
        return condensed


def check_molecule_grid(molecule, grid):
    """
    Check whether molecular grid is valid or not.

    Parameters
    ----------
    molecule: Molecule
        See `horton2_wrapper.wrapper.molecule.Molecule`.
    grid: MolecularGrid
        See `horton2_wrapper.wrapper.grid.MolecularGrid`.

    """
    if not np.max(abs(grid.coordinates - molecule.coordinates)) < 1.0e-6:
        raise ValueError("Argument molecule & grid should have the same coordinates.")
    if not np.max(abs(grid.numbers - molecule.numbers)) < 1.0e-6:
        raise ValueError("Arguments molecule & grid should have the same numbers.")
    if not np.max(abs(grid.pseudo_numbers - molecule.pseudo_numbers)) < 1.0e-6:
        raise ValueError("Arguments molecule & grid should have the same pseudo_numbers.")


def compute_molecular_multipole_moment(mol, l=1, func_type="cart"):
    """
    Compute molecular multipole moment on AO basis.

    Parameters
    ----------
    mol: Molecule
        See `horton2_wrapper.wrapper.molecule.Molecule`
    l: int, default=1
        Angular moment index. Default is dipole moment.
    func_type: {'cart', 'sph'}
        The moment type, Cartesian or Spherical.

    Returns
    -------
    np.ndarray, shape=(M, N, N)
        `M` components of multipole moment with `l` and `N` is the number of basis functions.

    """
    moments_ao = []
    for xyz in iter_cartesian_powers(l):
        moment = mol.lf.create_two_index()
        mol.obasis.compute_multipole_moment(np.asarray(xyz), np.array([0.0, 0.0, 0.0]), moment)
        moments_ao.append(moment._array)
    if func_type == "sph":
        # TODO: transfer it to real spherical harmonics
        pass
    return np.asarray(moments_ao)


def iter_cartesian_powers(order):
    """Iterate over Cartesian powers in `alphabetical` order"""
    for nx in range(order, -1, -1):
        for ny in range(order - nx, -1, -1):
            yield nx, ny, order - nx - ny

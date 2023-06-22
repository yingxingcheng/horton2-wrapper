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
"""Grid Wrapper Module."""

import numpy as np
from horton import BeckeMolGrid, solve_poisson_becke

from horton2_wrapper.molecule import Molecule

__all__ = ["MolecularGrid", "solve_atomic_poisson_beck"]


class MolecularGrid(BeckeMolGrid):
    """Becke-Lebedev molecular grid for numerical integrations."""

    def __init__(
        self, coordinates, numbers, pseudo_numbers, specification="medium", k=3, rotate=False
    ):
        """Initialize class.
        Parameters
        ----------
        coordinates : np.ndarray, shape=(M, 3)
            Cartesian coordinates of `M` atoms in the molecule.
        numbers : np.ndarray, shape=(M,)
            Atomic number of `M` atoms in the molecule.
        pseudo_numbers : np.ndarray, shape=(M,)
            Pseudo-number of `M` atoms in the molecule.
        specification : str, optional
            Specification of grid. Either choose from ['coarse', 'medium', 'fine', 'veryfine',
            'ultrafine', 'insane'] or provide a string of 'rname:rmin:rmax:nrad:nang' format.
            Here 'rname' denotes the type of radial grid and can be chosen from ['linear', 'exp',
            'power'], 'rmin' and 'rmax' specify the first and last radial grid points in angstrom,
            'nrad' specify the number of radial grid points, and 'nang' specify the number of
            angular Lebedev-Laikov grid. The 'nang' can be chosen from (6, 14, 26, 38, 50, 74, 86,
            110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030,
            2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810).
        k : int, optional
            The order of the switching function in Becke's weighting scheme.
        rotate : bool, optional
            Whether to randomly rotate spherical grids.
        """
        self._coordinates = coordinates
        self._numbers = numbers
        self._pseudo_numbers = pseudo_numbers
        self._k = k
        self._rotate = rotate
        self.specification = specification

        super(MolecularGrid, self).__init__(
            coordinates,
            numbers,
            pseudo_numbers,
            agspec=specification,
            k=k,
            random_rotate=rotate,
            mode="keep",
        )

    @classmethod
    def from_molecule(cls, molecule, specification="medium", k=3, rotate=False):
        """Initialize the class given an instance of Molecule.
        Parameters
        ----------
        molecule : instance of Molecule
            Instance of Molecule class.
        specification : str, optional
            Specification of grid. Either choose from ['coarse', 'medium', 'fine', 'veryfine',
            'ultrafine', 'insane'] or provide a string of 'rname:rmin:rmax:nrad:nang' format.
            Here 'rname' denotes the type of radial grid and can be chosen from ['linear', 'exp',
            'power'], 'rmin' and 'rmax' specify the first and last radial grid points in angstrom,
            'nrad' specify the number of radial grid points, and 'nang' specify the number of
            angular Lebedev-Laikov grid. The 'nang' can be chosen from (6, 14, 26, 38, 50, 74, 86,
            110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030,
            2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810).
        k : int, optional
            The order of the switching function in Becke's weighting scheme.
        rotate : bool, optional
            Whether to randomly rotate spherical grids.
        """
        if not isinstance(molecule, Molecule):
            raise TypeError("Argument molecule should be an instance of Molecule class.")
        coords, nums, pnums = molecule.coordinates, molecule.numbers, molecule.pseudo_numbers
        return cls(coords, nums, pnums, specification, k, rotate)

    @classmethod
    def from_file(cls, fname, specification="medium", k=3, rotate=False):
        """Initialize the class given an instance of Molecule.
        Parameters
        ----------
        fname : str
            Path to molecule's files.
        specification : str, optional
            Specification of grid. Either choose from ['coarse', 'medium', 'fine', 'veryfine',
            'ultrafine', 'insane'] or provide a string of 'rname:rmin:rmax:nrad:nang' format.
            Here 'rname' denotes the type of radial grid and can be chosen from ['linear', 'exp',
            'power'], 'rmin' and 'rmax' specify the first and last radial grid points in angstrom,
            'nrad' specify the number of radial grid points, and 'nang' specify the number of
            angular Lebedev-Laikov grid. The 'nang' can be chosen from (6, 14, 26, 38, 50, 74, 86,
            110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030,
            2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810).
        k : int, optional
            The order of the switching function in Becke's weighting scheme.
        rotate : bool, optional
            Whether to randomly rotate spherical grids.
        """
        mol = Molecule.from_file(fname)
        return cls.from_molecule(mol, specification, k, rotate)

    @property
    def coordinates(self):
        """Cartesian coordinates of atomic centers."""
        return self._coordinates

    @property
    def numbers(self):
        """Atomic number of atomic centers."""
        return self._numbers

    @property
    def pseudo_numbers(self):
        """Pseudo atomic number of atomic centers."""
        return self._pseudo_numbers

    @property
    def npoints(self):
        """Number of grid points."""
        return self.points.shape[0]

    @property
    def points(self):
        """Cartesian coordinates of grid points."""
        return super(MolecularGrid, self).points

    @property
    def weights(self):
        """Integration weight of grid points."""
        return super(MolecularGrid, self).weights

    def integrate(self, *args, **kwargs):
        """Integrate the property evaluated on the grid points.
        Parameters
        ----------
        value : np.ndarray
           Property value evaluated on the grid points.
        """
        return super(MolecularGrid, self).integrate(*args, **kwargs)


def solve_atomic_poisson_beck(grid, rho, nbatom, lmax=4):
    """Solve atomic poisson beck equation.

    Parameters
    ----------
    grid:
        Molecular grid.
    rho: np.ndarray
        Density on molecular grid.
    nbatom:
        The number of atoms in the molecule.
    lmax:
        The maximum angular momentum to consider when computing multipole moments.

    Returns
    -------
    List:
        A list of atomic potentials.
    """
    pots = np.zeros(grid.size)
    begin = 0
    for i in range(nbatom):
        atgrid = grid.subgrids[i]
        end = begin + atgrid.size
        becke_weights = grid.becke_weights[begin:end]
        density_decomposition = atgrid.get_spherical_decomposition(
            rho[begin:end], becke_weights, lmax=lmax
        )
        hartree_decomposition = solve_poisson_becke(density_decomposition)
        grid.eval_decomposition(hartree_decomposition, atgrid.center, pots)
        begin = end
    return pots

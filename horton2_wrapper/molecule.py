from __future__ import print_function, division
import numpy as np
from horton import (
    DenseLinalgFactory,
    IOData,
    get_gobasis,
    compute_nucnuc,
    periodic,
)

__all__ = ["Molecule"]


class Molecule(object):
    """
    Molecular object used in SCF calculation.
    """

    def __init__(self, coordinates, numbers, pseudo_numbers, basis, spin, charges):
        """
        Initialize function.

        Parameters
        ----------
        coordinates: np.ndarray
            The coordinates of the molecule in atomic unit.
        numbers: np.ndarray
            An array of atomic number of the molecule.
        pseudo_numbers: np.ndarray
            An array of atomic pseudo charges of the molecule.
        basis: basestring
            The name of basis sets used in SCF calculations.
        spin: int
            Spin multiplicity :math:`2S+1`.
        charges
        """
        self.coordinates = coordinates
        self.numbers = numbers
        self.pseudo_numbers = pseudo_numbers
        self.basis = basis
        self.spin = spin
        self.charges = charges
        # self.grid_params = grid_params or {"mode": "keep"}
        # self.grid_params["mode"] = "keep"

        self.cache = {}
        self._obasis = None
        self._lf = None

    @property
    def atomic_symbol(self):
        """All atomic symbols of the molecule."""
        return [periodic[int(i)].symbol for i in self.pseudo_numbers]

    @property
    def atomic_Z(self):
        """All atomic numbers of the molecule."""
        # TODO: check, are pseudo_numbers or numbers the atomic numbers?
        # return [int(i) for i in self.pseudo_numbers]
        return [int(i) for i in self.numbers]

    @property
    def lf(self):
        """Linear factory."""
        self._lf = self._lf or DenseLinalgFactory(self.obasis.nbasis)
        return self._lf

    @property
    def obasis(self):
        """A GTO basis object."""
        self._obasis = self._obasis or get_gobasis(self.coordinates, self.numbers, self.basis)
        return self._obasis

    @property
    def nbasis(self):
        """Number of basis functions."""
        return self.obasis.nbasis

    @property
    def natom(self):
        """Number of atoms in the molecule."""
        return len(self.numbers)

    @property
    def nelectrons(self):
        """Number of electrons in the molecule."""
        return int(np.sum(self.pseudo_numbers))

    @property
    def olp(self):
        """Overlap integral in AO basis."""
        if "int_olp" not in self.cache:
            self.cache["int_olp"] = self.obasis.compute_overlap(self.lf)
        return self.cache["int_olp"]

    @property
    def kin(self):
        """Kinetic integral in AO basis."""
        if "int_kin" not in self.cache:
            self.cache["int_kin"] = self.obasis.compute_kinetic(self.lf)
        return self.cache["int_kin"]

    @property
    def na(self):
        """Nuclear attraction integral in AO basis"""
        if "int_na" not in self.cache:
            self.cache["int_na"] = self.obasis.compute_nuclear_attraction(
                self.coordinates, self.pseudo_numbers, self.lf
            )
        return self.cache["int_na"]

    @property
    def nn(self):
        """Nuclear attraction integral in AO basis."""
        if "int_nn" not in self.cache:
            self.cache["int_nn"] = compute_nucnuc(self.coordinates, self.pseudo_numbers)
        return self.cache["int_nn"]

    @property
    def eri(self):
        """Electron repulsion integral in AO basis using physics notation."""
        if "int_eri" not in self.cache:
            self.cache["int_eri"] = self.obasis.compute_electron_repulsion(self.lf)
        return self.cache["int_eri"]

    def eri_erf(self, mu):
        """Long-range ERI defined by Erf on AO basis."""
        key = "int_eri_ao_{:.3f}".format(mu)
        if key not in self.cache:
            self.cache[key] = self.obasis.compute_erf_repulsion(self.lf, mu=mu)
        return self.cache[key]

    @classmethod
    def from_file(cls, filename, basis, spin=1, charges=0):
        """Build molecule from a file."""
        return cls.from_iodata(IOData.from_file(filename), basis, spin, charges)

    @classmethod
    def from_iodata(cls, data, basis, spin=1, charges=0):
        """
        Construct from IOData.

        If a GOBasis object is included in IOData, then we assume that the DFT calculation is
        finished, and read info from the IOData. Otherwise, a new DFT object is constructed with
        `is_done` being False.

        Parameters
        ----------
        data: IOData
           See `horton.IOData`.
        basis:
            Basis sets.
        spin:
            The spin of molecule.
        charges:
            The charges of molecule.

        Returns
        -------
        DFT

        """
        obj = cls(data.coordinates, data.numbers, data.pseudo_numbers, basis, spin, charges)
        if hasattr(data, "lf"):
            obj._lf = getattr(data, "lf")
        if hasattr(data, "obasis"):
            obj._obasis = getattr(data, "obasis")
        return obj

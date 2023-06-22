from __future__ import division, print_function

import numpy as np
import pytest
from horton2_wrapper import *
from tests.common import *

# from angstrom to atomic unit
ang2au = 1.8897259886


def test_molecule_h2o():
    mol = Molecule.from_iodata(get_h2o(), "aug-cc-pVDZ")
    coords_ref = np.array(
        [
            [0.783837, -0.492236, -0.000000],
            [-0.000000, 0.062020, -0.000000],
            [-0.783837, -0.492236, -0.000000],
        ]
    )

    assert mol.atomic_symbol == ["H", "O", "H"]
    assert mol.coordinates == pytest.approx(coords_ref * ang2au)
    assert None not in [mol.lf, mol.obasis]
    assert mol.atomic_Z == [1, 8, 1]
    assert mol.natom == 3


def test_molecule_h2():
    mol = Molecule.from_iodata(get_h2(), "aug-cc-pVDZ")
    coords_ref = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.098]])

    assert mol.atomic_symbol == ["H", "H"]
    assert mol.coordinates == pytest.approx(coords_ref * ang2au)
    assert None not in [mol.lf, mol.obasis]
    assert mol.atomic_Z == [1, 1]
    assert mol.natom == 2

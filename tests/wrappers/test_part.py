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

import numpy as np
import pytest

from horton2_wrapper import *
from tests.common import *

np.set_printoptions(precision=10, linewidth=200, suppress=True, threshold=np.inf)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_iter_cartesian_powers(order):
    res_ref = {
        0: [(0, 0, 0)],
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
    }
    for xyz, ref in zip(iter_cartesian_powers(order), res_ref[order]):
        assert xyz == ref


@pytest.mark.parametrize("l", [0, 1, 2])
def test_compute_moleuclar_multipole_moment(l):
    mol = Molecule.from_iodata(get_h2o(), basis="cc-pVDZ")
    res = compute_molecular_multipole_moment(mol, l)
    assert res.shape == ((l + 1) * (l + 2) // 2, mol.nbasis, mol.nbasis)


def test_dens_part():
    dft = DFT.from_file(load_fchk("water1.fchk"), xc="pbe")
    dp = DensPart(dft)

    res = dp.compute_aim_transition_multipole_moment(1)
    assert res.shape == (12, dft.mol.nbasis, dft.mol.nbasis)

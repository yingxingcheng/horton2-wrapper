import numpy as np
import pytest
from horton import load_h5
from horton2_wrapper import *
from tests.common import *


def test_hf_to_file():
    mol = Molecule.from_iodata(data=get_h2o(), basis="6-31g(d)")
    hf = HF(mol=mol)
    assert hf.is_done is False
    hf.run()
    assert hf.is_done is True

    saved_keys = [
        "lf",
        "coordinates",
        "obasis",
        "numbers",
        "pseudo_numbers",
        "exp_alpha",
    ]

    with tmpfile(".h5") as fname:
        print(fname)
        hf.to_file(fname)
        iodata = load_h5(fname)
        for k in saved_keys:
            assert k in iodata
        for k, v in iodata.items():
            if "energy" in k:
                assert v == pytest.approx(hf.ham.cache[k], 1e-10)


@pytest.mark.parametrize("sc_type", ["plain", "cdiis"])
def test_hf(sc_type):
    mol = Molecule.from_iodata(data=get_h2o(), basis="cc-pVDZ")
    hf = HF(mol=mol)
    assert hf.is_done is False
    hf.run(sc_type)
    assert hf.is_done is True

    rt_previous = {
        "energy": -76.025896285286194,
        "exp_alpha": np.array(
            [
                -20.548047280389707,
                -1.3313877663726761,
                -0.70683717499898535,
                -0.55595762656724068,
                -0.49108019506479106,
                0.18591763000506215,
                0.25527900850832308,
                0.8054303226321432,
                0.82856232585934586,
                1.1614308644623972,
                1.2017228377672824,
                1.2482985386850212,
                1.460360693421848,
                1.4872252759600568,
                1.7005288811421662,
                1.8795199745217244,
                1.9039035561567033,
                2.4628162224448382,
                2.4753585904755169,
                3.2563466493320208,
                3.3553130136515197,
                3.4726454116323171,
                3.9110158964835904,
                4.1133159006021769,
            ]
        ),
        "hartree": 46.890326866235284,
        "kin": 75.97075998377996,
        "ne": -199.07032586657695,
        "nn": 9.1571750364299866,
        "x_hf": -8.973832305154477,
    }

    abs = 1e-4
    for k, v in rt_previous.items():
        if k == "exp_alpha":
            assert getattr(hf, k).energies == pytest.approx(v, abs=abs)
        elif k == "energy":
            assert getattr(hf, k) == pytest.approx(v, abs=abs)
        else:
            assert hf.ham.cache["energy_" + k] == pytest.approx(v, abs=abs)

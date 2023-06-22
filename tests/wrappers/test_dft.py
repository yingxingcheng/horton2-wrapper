from __future__ import division, print_function

import numpy as np
import pytest
from horton import load_h5

from horton2_wrapper import *
from tests.common import *

np.set_printoptions(precision=6, linewidth=200, suppress=True, threshold=np.inf)


def test_dft_exx_energy():
    mol = Molecule.from_iodata(data=get_h2o(), basis="6-31g(d)")
    dft = DFT(mol=mol, xc="pbe")
    assert dft.is_done is False
    dft.run()
    assert dft.is_done is True
    e_exx = dft.compute_exx()
    dft_x_e = dft.ham.cache["energy_libxc_gga_x_pbe"]

    assert e_exx == pytest.approx(-8.95344505409, 1e-5)
    assert dft_x_e == pytest.approx(-8.93864828791, 1e-5)


def test_dft_to_file():
    mol = Molecule.from_iodata(get_h2o(), basis="6-31g(d)")
    dft = DFT(mol, xc="lda")
    assert dft.is_done is False
    dft.run()
    assert dft.is_done is True

    saved_keys = [
        "lf",
        "coordinates",
        "obasis",
        "numbers",
        "pseudo_numbers",
        "exp_alpha",
    ]

    with tmpfile(".h5") as fname:
        dft.to_file(fname)

        # check energy
        ioh5 = load_h5(fname)
        for k in saved_keys:
            assert k in ioh5

        for k, v in ioh5.items():
            if "energy" in k:
                assert v == pytest.approx(dft.ham.cache[k], 1e-10)


def test_dft_lda():
    mol = Molecule.from_iodata(get_h2o(), basis="6-31g(d)")
    dft = DFT(mol, xc="lda")
    assert dft.is_done is False
    dft.run()
    assert dft.is_done is True

    rt_previous = {
        "energy": -75.840489821694746,
        "exp_alpha": np.array(
            [
                -18.58230481517597,
                -0.89645482184183045,
                -0.47394533974322334,
                -0.29938706464077286,
                -0.22841663753724872,
                0.045271491553249367,
                0.12825353255457303,
                0.76332895055364547,
                0.80243034890203158,
                0.83062512325223159,
                0.86463348691682496,
                1.0042272336474725,
                1.3160465964155548,
                1.6707142466917595,
                1.6768134032677695,
                1.718722082489359,
                2.2334850292684449,
                2.520278714219923,
            ]
        ),
        "grid": -8.779901102250568,
        "hartree": 46.821068827153724,
        "kin": 75.86645763295653,
        "ne": -198.90529021598442,
        "nn": 9.1571750364299866,
    }
    for k, v in rt_previous.items():
        if k == "exp_alpha":
            assert getattr(dft, k).energies == pytest.approx(v, abs=1e-4)
        elif k == "grid":
            assert dft.ham.cache["energy_" + k + "_group"] == pytest.approx(v, abs=1e-4)
        elif k == "energy":
            assert getattr(dft, k) == pytest.approx(v, abs=1e-4)
        else:
            assert dft.ham.cache["energy_" + k] == pytest.approx(v, abs=1e-4)


def test_dft_pbe():
    mol = Molecule.from_iodata(get_h2o(), basis="6-31g(d)")
    dft = DFT(mol, xc="pbe")
    assert dft.is_done is False
    dft.run()
    assert dft.is_done is True

    rt_previous = {
        "energy": -76.319081513896819,
        "exp_alpha": np.array(
            [
                -18.741680514615258,
                -0.90374899607804215,
                -0.47273367440296421,
                -0.29901863797777328,
                -0.22546422195580798,
                0.047093994157163643,
                0.12992980227665973,
                0.7542183804262228,
                0.79451255354023687,
                0.84502543732431601,
                0.87439499193454917,
                1.0199047335161955,
                1.3209866823333651,
                1.6703806455245105,
                1.6760857663647439,
                1.7170769473238023,
                2.2300682992916969,
                2.5143154025366314,
            ]
        ),
        "grid": -9.267847445230666,
        "hartree": 46.84547411400169,
        "kin": 75.98754611390268,
        "ne": -199.0414293330005,
        "nn": 9.1571750364299866,
    }

    for k, v in rt_previous.items():
        if k == "exp_alpha":
            assert getattr(dft, k).energies == pytest.approx(v, abs=1e-4)
        elif k == "grid":
            assert dft.ham.cache["energy_" + k + "_group"] == pytest.approx(v, abs=1e-4)
        elif k == "energy":
            assert getattr(dft, k) == pytest.approx(v, abs=1e-4)
        else:
            assert dft.ham.cache["energy_" + k] == pytest.approx(v, abs=1e-4)


def test_dft_b3lyp():
    mol = Molecule.from_iodata(get_h2o(), basis="6-31g(d)")
    dft = DFT(mol, xc="b3lyp")
    print(dft)
    print("-" * 80)
    assert dft.is_done is False
    dft.run()
    assert dft.is_done is True
    print(dft)
    print("-" * 80)

    rt_previous = {
        "energy": -76.406156776346975,
        "exp_alpha": np.array(
            [
                -19.12494652215198,
                -0.99562109649344044,
                -0.52934359625260619,
                -0.35973919172781244,
                -0.28895110439599314,
                0.068187099284877942,
                0.1532902668612677,
                0.80078130036326101,
                0.84958389626115138,
                0.89305132504935913,
                0.92182191946355896,
                1.074508959522454,
                1.3767806620540104,
                1.7405943781554678,
                1.7462666980125516,
                1.7861275433424106,
                2.3057917944397714,
                2.5943014303914662,
            ]
        ),
        "grid": -7.568923843396495,
        "hartree": 46.893530019953076,
        "kin": 76.03393036526309,
        "ne": -199.129803256826,
        "nn": 9.1571750364299866,
        "x_hf": -1.792065097770653,
    }

    for k, v in rt_previous.items():
        if k == "exp_alpha":
            assert getattr(dft, k).energies == pytest.approx(v, abs=1e-4)
        elif k == "grid":
            assert dft.ham.cache["energy_" + k + "_group"] == pytest.approx(v, abs=1e-4)
        elif k == "energy":
            assert getattr(dft, k) == pytest.approx(v, abs=1e-4)
        else:
            assert dft.ham.cache["energy_" + k] == pytest.approx(v, abs=1e-4)


def test_dft_pbe0():
    mol = Molecule.from_iodata(get_h2o(), basis="6-31g(d)")
    dft = DFT(mol, xc="pbe0_13")
    print(dft)
    print("-" * 80)
    assert dft.is_done is False
    dft.run()
    assert dft.is_done is True
    print(dft)
    print("-" * 80)

    assert dft.xc_funcs[0].get_exx_fraction() == pytest.approx(1 / 3)

    rt_previous = {
        "energy": -76.3246339897,
        "exp_alpha": np.array(
            [
                -19.350613,
                -1.061306,
                -0.566309,
                -0.399364,
                -0.32817,
                0.099171,
                0.184902,
                0.842784,
                0.898494,
                0.943998,
                0.969598,
                1.129485,
                1.419137,
                1.774294,
                1.780077,
                1.820446,
                2.351241,
                2.643646,
            ]
        ),
        "grid": -6.28992025264,
        "hartree": 46.8652109307,
        "kin": 75.9914878378,
        "ne": -199.062392458,
        "nn": 9.1571750364299866,
        "x_hf": -2.98619508407,
    }

    for k, v in rt_previous.items():
        if k == "exp_alpha":
            assert getattr(dft, k).energies == pytest.approx(v, abs=1e-4)
        elif k == "grid":
            assert dft.ham.cache["energy_" + k + "_group"] == pytest.approx(v, abs=1e-4)
        elif k == "energy":
            assert getattr(dft, k) == pytest.approx(v, abs=1e-4)
        else:
            assert dft.ham.cache["energy_" + k] == pytest.approx(v, abs=1e-4)


def test_dft_tpss():
    mol = Molecule.from_iodata(get_h2o(), basis="6-31g(d)")
    dft = DFT(mol, xc="tpss")
    assert dft.is_done is False
    dft.run()
    assert dft.is_done is True

    rt_previous = {
        "energy": -76.407936399996032,
        "exp_alpha": np.array(
            [
                -18.88140274471057,
                -0.92689455671943777,
                -0.48003995365965246,
                -0.3068412737938469,
                -0.23305348781949642,
                0.055765650587591052,
                0.13876252429050115,
                0.78022341323374989,
                0.82177783193535281,
                0.86268998494640037,
                0.89541154561880765,
                1.0425268139841128,
                1.3431492859747944,
                1.7062096596569536,
                1.7110987254283361,
                1.7506809108152641,
                2.2882571842385735,
                2.5849159392493357,
            ]
        ),
        "grid": -9.361522808603524,
        "hartree": 46.86311076030199,
        "kin": 76.04019446027614,
        "ne": -199.10689384840063,
        "nn": 9.1571750364299866,
    }

    abs = 5e-3
    for k, v in rt_previous.items():
        if k == "exp_alpha":
            assert getattr(dft, k).energies == pytest.approx(v, abs=abs)
        elif k == "grid":
            assert dft.ham.cache["energy_" + k + "_group"] == pytest.approx(v, abs=abs)
        elif k == "energy":
            assert getattr(dft, k) == pytest.approx(v, abs=abs)
        else:
            assert dft.ham.cache["energy_" + k] == pytest.approx(v, abs=abs)


def test_dft_m05():
    mol = Molecule.from_iodata(get_h2o(), basis="6-31g(d)")
    dft = DFT(mol, xc="m05")
    assert dft.is_done is False
    dft.run()
    assert dft.is_done is True

    rt_previous = {
        "energy": -76.372223106410885,
        "exp_alpha": np.array(
            [
                -19.174675917533499,
                -1.0216889289766689,
                -0.54324149010045464,
                -0.37631403914157158,
                -0.30196183487620326,
                0.079896573985756419,
                0.16296304612701332,
                0.81419059490960388,
                0.86377461055569127,
                0.9243929453024935,
                0.95050094195149326,
                1.1033737076332981,
                1.4108569929549999,
                1.7561523962868733,
                1.761532111350379,
                1.8055689722633752,
                2.3348442517458823,
                2.6275437456471868,
            ]
        ),
        "grid": -6.821114560989138,
        "hartree": 46.93245844915478,
        "kin": 76.05549816546615,
        "ne": -199.18635862588496,
        "nn": 9.1571750364299866,
        "x_hf": -2.50988157058769,
    }

    abs = 5e-2
    for k, v in rt_previous.items():
        if k == "exp_alpha":
            assert getattr(dft, k).energies == pytest.approx(v, abs=abs)
        elif k == "grid":
            assert dft.ham.cache["energy_" + k + "_group"] == pytest.approx(v, abs=abs)
        elif k == "energy":
            assert getattr(dft, k) == pytest.approx(v, abs=abs)
        else:
            assert dft.ham.cache["energy_" + k] == pytest.approx(v, abs=abs)


@pytest.mark.parametrize("fn", ["water1.fchk"])
def test_dft_from_file(fn):
    dft = DFT.from_file(filename=load_fchk(fn), xc="pbe")
    assert dft.is_done is True
    e_ref = dft.energy
    calc_res = dft.reproduce_result()
    assert e_ref == pytest.approx(calc_res["energy"], abs=1e-4)


@pytest.mark.parametrize("fn", ["h2_svwn3_rpa.fchk"])
def test_dft_gaussian_lsda(fn):
    dft = DFT.from_file(load_fchk(fn))
    print(dft)
    print("-" * 80)
    assert dft.is_done is True
    print("LSDA (SVWN3) from gaussian: {}".format(dft.energy))
    assert dft.energy == pytest.approx(-1.16812504976, abs=1e-9)

    print("DFT based on Horton: ")
    for xc in ["svwn1", "svwn2", "svwn3", "svwn4", "svwn5", "svwn5_rpa"]:
        mol = Molecule.from_file(load_fchk(fn), basis="aug-cc-pVDZ")
        dft2 = DFT(mol, xc=xc)
        dft2.run(threshold=1e-9)
        print("{}: {}".format(xc, dft2.energy))
        assert not dft.energy == pytest.approx(dft2.energy, abs=1e-8)


@pytest.mark.parametrize("fn", ["water_svwn5.fchk"])
def test_dft_svwn5(fn):
    dft = DFT.from_file(load_fchk(fn))
    print(dft)
    print("-" * 80)
    assert dft.is_done is True
    print("DFT with SVWN5 from gaussian: {}".format(dft.energy))
    assert dft.energy == pytest.approx(-75.9057083067, abs=1e-9)

    print("DFT based on Horton: ")
    for xc in ["svwn5"]:
        mol = Molecule.from_file(load_fchk(fn), basis="aug-cc-pVTZ")
        dft2 = DFT(mol, xc=xc)
        dft2.run(threshold=1e-9)
        print("{}: {}".format(xc, dft2.energy))
        assert dft.energy == pytest.approx(dft2.energy, abs=1e-4)

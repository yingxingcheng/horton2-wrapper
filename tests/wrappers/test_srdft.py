import numpy as np
import pytest
from horton import load_h5
from horton2_wrapper import *
from tests.common import *


@pytest.mark.parametrize("basis", ["6-31g(d)"])
@pytest.mark.parametrize("mu", [0.4])
@pytest.mark.parametrize("xc", ["srpbe"])
def test_hfdft_exx_energy(basis, mu, xc):
    mol = Molecule.from_iodata(get_h2o(), basis=basis)
    dft = HFsrDFT(mol, mu=mu, xc=xc)
    dft.run()
    e_exx = dft.compute_exx()
    dft_x_e = dft.ham.cache["energy_libxc_sr_gga_x_pbe_erfgws"]

    assert e_exx == pytest.approx(-8.95800378896, 1e-5)
    assert dft_x_e == pytest.approx(-6.92424309047, 1e-5)


@pytest.mark.parametrize("basis", ["6-31g(d)"])
def test_hfsrdft_to_file(basis):
    mol = Molecule.from_iodata(get_h2o(), basis=basis)
    dft = HFsrDFT(mol, mu=0.4, xc="srlda")
    dft.run()

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
        iodata = load_h5(fname)
        for k in saved_keys:
            assert k in iodata.keys()

        for k, v in iodata.items():
            if "energy" in k:
                assert v == pytest.approx(dft.ham.cache[k], 1e-10)


@pytest.mark.parametrize("basis", ["cc-pVDZ"])
@pytest.mark.parametrize("mu", [10])
def test_srdft_lrc_lda(basis, mu):
    mol = Molecule.from_iodata(get_h2o(), basis=basis)
    srdft = HFsrDFT(mol, mu=mu, xc="lrclda")
    srdft.run()

    e_ref_dict = {}
    e_ref_dict[10] = {
        "energy": -76.688281194248,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -20.58572734,
                -1.38741979,
                -0.76240626,
                -0.61116666,
                -0.54619223,
                0.15342530,
                0.22484386,
                0.75691660,
                0.78039944,
                1.11273602,
                1.15204298,
                1.20293424,
                1.40943699,
                1.43854242,
                1.65033789,
                1.83087607,
                1.85449539,
                2.41061801,
                2.42294890,
                3.20150644,
                3.30056916,
                3.41841759,
                3.85699325,
                4.05994122,
            ]
        ),
    }

    if np.isclose(mu, 1e-6):
        abs = 1e-3
    else:
        abs = 1e-4
    for k, v in e_ref_dict[mu].items():
        if k == "exp_alpha":
            assert getattr(srdft, k).energies == pytest.approx(v, abs=abs)
        else:
            assert srdft.ham.cache[k] == pytest.approx(v, abs=abs)


@pytest.mark.parametrize("mol", [get_h2o()])
@pytest.mark.parametrize("basis", ["cc-pVDZ"])
@pytest.mark.parametrize("mu", [1e-6, 0.4, 10, 1000])
def test_srdft_sr_lda_pw92(mol, basis, mu):
    mol = Molecule.from_iodata(get_h2o(), basis=basis)
    srdft = HFsrDFT(mol, mu=mu, xc="sr_lda_pw92")
    print(srdft)
    print(srdft.is_done)
    srdft.run()
    print(srdft.is_done)

    e_ref_dict = {}
    e_ref_dict[1e-6] = {
        "energy": -75.851170031654,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -18.57711043,
                -0.89054161,
                -0.46582795,
                -0.29408068,
                -0.22631231,
                0.03327290,
                0.10921437,
                0.53969081,
                0.55576990,
                0.83644172,
                0.85654294,
                0.93277426,
                1.15944130,
                1.18990154,
                1.39369843,
                1.54511260,
                1.57966726,
                2.04962702,
                2.07087429,
                2.81317170,
                2.89840169,
                3.01563229,
                3.42660441,
                3.61682467,
            ]
        ),
    }
    e_ref_dict[0.4] = {
        "energy": -75.917924063746,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -18.79300939,
                -1.10819278,
                -0.64395511,
                -0.47497292,
                -0.40972309,
                0.15342727,
                0.22389947,
                0.69480152,
                0.71426730,
                1.01574734,
                1.03994628,
                1.10241176,
                1.33661158,
                1.36364672,
                1.57231072,
                1.72424948,
                1.76378428,
                2.24265373,
                2.26478269,
                3.01622313,
                3.10367827,
                3.22002364,
                3.63011866,
                3.82040632,
            ]
        ),
    }
    e_ref_dict[10] = {
        "energy": -76.064594950596,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -20.53393387,
                -1.33185192,
                -0.70753345,
                -0.55670827,
                -0.49194185,
                0.18579290,
                0.25509333,
                0.80503111,
                0.82812644,
                1.16070411,
                1.20069163,
                1.24775251,
                1.46013935,
                1.48707832,
                1.70031792,
                1.87947521,
                1.90367612,
                2.46201160,
                2.47476318,
                3.25596422,
                3.35493275,
                3.47225326,
                3.91046721,
                4.11276948,
            ]
        ),
    }
    e_ref_dict[1000] = {
        "energy": -76.025903843778,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -20.54804666,
                -1.33138620,
                -0.70683614,
                -0.55595581,
                -0.49107828,
                0.18591792,
                0.25527913,
                0.80543086,
                0.82856284,
                1.16143210,
                1.20172441,
                1.24829949,
                1.46036112,
                1.48722591,
                1.70052930,
                1.87952109,
                1.90390376,
                2.46281723,
                2.47535959,
                3.25634848,
                3.35531487,
                3.47264714,
                3.91101761,
                4.11331768,
            ]
        ),
    }

    if np.isclose(mu, 1e-6):
        abs = 1e-3
    else:
        abs = 1e-4
    for k, v in e_ref_dict[mu].items():
        if k == "exp_alpha":
            assert getattr(srdft, k).energies == pytest.approx(v, abs=abs)
        else:
            assert srdft.ham.cache[k] == pytest.approx(v, abs=abs)


@pytest.mark.parametrize("basis", ["cc-pVDZ"])
@pytest.mark.parametrize("mu", [1e-6, 0.4, 10, 1000])
def test_srdft_srlda(basis, mu):
    mol = Molecule.from_iodata(get_h2o(), basis=basis)
    srdft = HFsrDFT(mol, mu=mu, xc="srlda")
    srdft.run()

    e_ref_dict = {}
    e_ref_dict[1e-6] = {
        "energy": -75.853990127928,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -18.57722000,
                -0.89071654,
                -0.46598697,
                -0.29423590,
                -0.22646599,
                0.03328341,
                0.10921514,
                0.53958444,
                0.55566414,
                0.83637133,
                0.85647118,
                0.93272272,
                1.15932716,
                1.18980911,
                1.39359551,
                1.54501055,
                1.57957875,
                2.04950209,
                2.07074457,
                2.81299291,
                2.89822487,
                3.01546371,
                3.42643673,
                3.61666596,
            ]
        ),
    }
    e_ref_dict[0.4] = {
        "energy": -75.920713767762,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -18.79311813,
                -1.10836686,
                -0.64411887,
                -0.47513138,
                -0.40987884,
                0.15341029,
                0.22387190,
                0.69467902,
                0.71414859,
                1.01567275,
                1.03986932,
                1.10235348,
                1.33650871,
                1.36355746,
                1.57221553,
                1.72414967,
                1.76370312,
                2.24253098,
                2.26465555,
                3.01604558,
                3.10350318,
                3.21985491,
                3.62994900,
                3.82024306,
            ]
        ),
    }
    e_ref_dict[10] = {
        "energy": -76.064696006598,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -20.53402332,
                -1.33185468,
                -0.70753305,
                -0.55670815,
                -0.49194155,
                0.18579256,
                0.25509340,
                0.80503129,
                0.82812633,
                1.16070395,
                1.20069144,
                1.24775249,
                1.46013960,
                1.48707800,
                1.70031814,
                1.87947418,
                1.90367627,
                2.46201162,
                2.47476264,
                3.25596492,
                3.35493344,
                3.47225391,
                3.91046770,
                4.11277001,
            ]
        ),
    }
    e_ref_dict[1000] = {
        "energy": -76.025903842665,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -20.54804666,
                -1.33138620,
                -0.70683614,
                -0.55595581,
                -0.49107828,
                0.18591792,
                0.25527913,
                0.80543086,
                0.82856284,
                1.16143210,
                1.20172441,
                1.24829949,
                1.46036112,
                1.48722591,
                1.70052930,
                1.87952109,
                1.90390376,
                2.46281723,
                2.47535959,
                3.25634848,
                3.35531487,
                3.47264714,
                3.91101761,
                4.11331768,
            ]
        ),
    }

    if np.isclose(mu, 1e-6):
        abs = 1e-3
    else:
        abs = 1e-4
    for k, v in e_ref_dict[mu].items():
        if k == "exp_alpha":
            assert getattr(srdft, k).energies == pytest.approx(v, abs=abs)
        else:
            assert srdft.ham.cache[k] == pytest.approx(v, abs=abs)


@pytest.mark.parametrize("basis", ["cc-pVDZ"])
@pytest.mark.parametrize("mu", [1e-6, 0.4, 10, 1000])
def test_srdft_srpbe(basis, mu):
    mol = Molecule.from_iodata(get_h2o(), basis=basis)
    srdft = HFsrDFT(mol, mu=mu, xc="srpbe")
    srdft.run()

    e_ref_dict = {}
    e_ref_dict[1e-6] = {
        "energy": -76.332501058160,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -18.73663094,
                -0.89794022,
                -0.46478741,
                -0.29383999,
                -0.22326761,
                0.03420486,
                0.10936510,
                0.53234239,
                0.54888619,
                0.84905482,
                0.87115643,
                0.94655271,
                1.16454981,
                1.19356271,
                1.39788972,
                1.54966634,
                1.58585627,
                2.05502104,
                2.07331643,
                2.80676905,
                2.89300207,
                3.01038944,
                3.42013224,
                3.61250006,
            ]
        ),
    }
    e_ref_dict[0.4] = {
        "energy": -76.335467059610,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -18.94993381,
                -1.11504859,
                -0.64284478,
                -0.47477311,
                -0.40811147,
                0.15432684,
                0.22597830,
                0.69274306,
                0.71235615,
                1.02063690,
                1.04592958,
                1.10879466,
                1.34050449,
                1.36581156,
                1.57570319,
                1.72501968,
                1.76777953,
                2.24688056,
                2.26622456,
                3.01341755,
                3.10137652,
                3.21757979,
                3.62625934,
                3.81774438,
            ]
        ),
    }
    e_ref_dict[10] = {
        "energy": -76.061738713538,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -20.53286106,
                -1.33180243,
                -0.70753933,
                -0.55671077,
                -0.49194941,
                0.18579775,
                0.25509132,
                0.80502992,
                0.82812868,
                1.16069694,
                1.20067818,
                1.24774539,
                1.46013885,
                1.48708822,
                1.70031788,
                1.87950162,
                1.90367564,
                2.46200237,
                2.47477023,
                3.25596591,
                3.35493447,
                3.47225492,
                3.91046750,
                4.11276877,
            ]
        ),
    }
    e_ref_dict[1000] = {
        "energy": -76.025903849973,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -20.54804667,
                -1.33138620,
                -0.70683614,
                -0.55595581,
                -0.49107828,
                0.18591792,
                0.25527913,
                0.80543086,
                0.82856284,
                1.16143210,
                1.20172441,
                1.24829949,
                1.46036112,
                1.48722591,
                1.70052930,
                1.87952109,
                1.90390376,
                2.46281723,
                2.47535959,
                3.25634848,
                3.35531487,
                3.47264714,
                3.91101761,
                4.11331768,
            ]
        ),
    }

    if np.isclose(mu, 1e-6):
        abs = 1e-3
    else:
        abs = 1e-4
    for k, v in e_ref_dict[mu].items():
        if k == "exp_alpha":
            assert getattr(srdft, k).energies == pytest.approx(v, abs=abs)
        else:
            assert srdft.ham.cache[k] == pytest.approx(v, abs=abs)


@pytest.mark.parametrize("mol", [get_h2o()])
@pytest.mark.parametrize("basis", ["cc-pVDZ"])
@pytest.mark.parametrize("mu", [10])
def test_srdft_lrc_pbe(mol, basis, mu):
    mol = Molecule.from_iodata(get_h2o(), basis=basis)
    dft = HFsrDFT(mol, mu=mu, xc="lrcpbe")
    dft.run()

    e_ref_dict = {}
    e_ref_dict[10] = {
        "energy": -76.357013928633,
        "energy_nn": 9.157175079691,
        "exp_alpha": np.array(
            [
                -20.53509785,
                -1.37505324,
                -0.74853995,
                -0.59662891,
                -0.53027207,
                0.17344686,
                0.24399749,
                0.77581893,
                0.79979257,
                1.13119541,
                1.17435093,
                1.21991141,
                1.41736376,
                1.45017140,
                1.66016498,
                1.84380865,
                1.86551355,
                2.42715235,
                2.43966175,
                3.21029443,
                3.30888118,
                3.42759345,
                3.87120915,
                4.07472382,
            ]
        ),
    }

    if np.isclose(mu, 1e-6):
        abs = 1e-3
    else:
        abs = 1e-4
    for k, v in e_ref_dict[mu].items():
        if k == "exp_alpha":
            assert getattr(dft, k).energies == pytest.approx(v, abs=abs)
        else:
            assert dft.ham.cache[k] == pytest.approx(v, abs=abs)


@pytest.mark.parametrize("data", [get_h2o(), get_n2(), get_h2()])
@pytest.mark.parametrize("basis", ["cc-pVDZ", "cc-pVTZ"])
def test_hfsrdft_hf_limit(data, basis):
    mol = Molecule.from_iodata(data, basis=basis)
    srdft_hf_limit = HFsrDFT(mol, mu=1e3, xc="srpbe")
    e2 = srdft_hf_limit.run()
    hf = HF(mol)
    e_hf = hf.run()
    assert e_hf == pytest.approx(e2, abs=1e-3)


@pytest.mark.parametrize("data", [get_h2o(), get_n2(), get_h2()])
@pytest.mark.parametrize("basis", ["cc-pVDZ", "cc-pVTZ"])
def test_hfsrdft_dft_limit(data, basis):
    mol = Molecule.from_iodata(data, basis=basis)
    dft = DFT(mol, xc="pbe")
    e_dft = dft.run()
    srdft_dft_limit = HFsrDFT(mol, mu=1e-6, xc="srpbe")
    e1 = srdft_dft_limit.run()
    assert e_dft == pytest.approx(e1, abs=1e-3)

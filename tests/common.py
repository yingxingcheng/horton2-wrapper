import os
import shutil
import tempfile
from contextlib import contextmanager
from horton import IOData, angstrom
import numpy as np
from horton2_wrapper import DensPart, Hardness

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


__all__ = [
    "get_h2",
    "get_h2o",
    "get_n2",
    "load_fchk",
    "load_data",
    "get_atom",
    "tmpdir",
    "tmpfile",
    "prepare_input_for_acks2_from_scf",
]


def get_h2o():
    """Get H2O molecule."""
    with path("lrccd.data", "water.xyz") as fpath:
        mol = IOData.from_file(str(fpath))
    return mol


def get_h2():
    """Get H2 molecule."""
    bond_length = 1.098 * angstrom
    mol = IOData(title="hydrogen")
    mol.coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]])
    mol.numbers = np.array([1, 1])
    return mol


def get_n2():
    """Get N2 molecule."""
    bond_length = 1.098 * angstrom
    mol = IOData(title="N2")
    mol.coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]])
    mol.numbers = np.array([7, 7])
    return mol


def get_atom(n):
    """Get H2 molecule."""
    mol = IOData(title="atom with Z = {}".format(n))
    mol.coordinates = np.array([[0.0, 0.0, 0.0]])
    mol.numbers = np.array([n])
    return mol


def load_fchk(fname):
    """Load fchk file."""
    with path("lrccd.data", fname) as fpath:
        filename = str(fpath)
    return filename


def load_data(name):
    """Load data file based on name."""
    fname = load_fchk("{}.fchk".format(name))
    return IOData.from_file(fname)


@contextmanager
def tmpdir(name):
    """Create temporary directory that gets deleted after accessing it."""
    dn = tempfile.mkdtemp(name)
    try:
        yield dn
    finally:
        shutil.rmtree(dn)


@contextmanager
def tmpfile(suffix):
    """Create temporary file that gets deleted after accessing it."""
    fn = tempfile.mktemp(suffix=suffix)
    try:
        yield fn
    finally:
        if os.path.exists(fn):
            os.remove(fn)


def prepare_input_for_acks2_from_scf(scf, lmax, dens_type="case_1"):
    """Prepare input for ACKS2.

    Parameters
    ----------
    scf: SCF
        The SCF object.
    lmax: int
        The maximum angular moment index.
    dens_type: {'case_1', 'case_2'}
        The type of density function, where `case_1` corresponds the distributed moment defined
        the ACKS2 model, while monopoles are removed in `case_2`.

    """
    dens_part = DensPart(scf)
    # chi_ks = ChiKS.from_denspart(dens_part, lmax, dens_type)
    hardness = Hardness(scf, chi_ks.F_ao)
    return chi_ks, hardness.eta, hardness.eta_h, hardness.eta_xc

import pytest
from hartree_fock import UHF
from hartree_fock.mix import DIIS, AlphaMixer
from quantum_systems import (
    TwoDimensionalHarmonicOscillator,
    construct_pyscf_system_ao,
)


@pytest.mark.skip
def get_pysf_uhf_energy(molecule, basis="ccpvdz", **kwargs):
    import pyscf

    mol = pyscf.gto.Mole()
    mol.build(atom=molecule, basis=basis, **kwargs)

    uhf = pyscf.scf.UHF(mol)

    return uhf.kernel() + mol.energy_nuc()


@pytest.mark.skip
def test_tdho_uhf():
    n = 2
    l = 12
    radius = 10
    num_grid_points = 401

    tdho = TwoDimensionalHarmonicOscillator(n, l, radius, num_grid_points)
    tdho.setup_system(add_spin=False, anti_symmetrize=False)

    uhf = UHF(tdho, verbose=True)

    uhf.compute_ground_state(tol=1e-10)

    assert abs(uhf.compute_energy() - 3.162691) < 1e-6


@pytest.mark.skip
def test_lithium_uhf():
    mol = "li"
    basis = "ccpvdz"
    kwargs = dict(spin=1)

    pyscf_energy = get_pysf_uhf_energy(mol, basis, **kwargs)
    system = construct_pyscf_system_ao(
        mol, basis, add_spin=False, anti_symmetrize=False, **kwargs
    )
    uhf = UHF(system, verbose=True)
    uhf.compute_ground_state(tol=1e-8)

    assert abs(uhf.compute_energy() - pyscf_energy) < 1e-6

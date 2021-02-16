from hartree_fock.rhf import RHF
from quantum_systems import TwoDimensionalHarmonicOscillator
from quantum_systems import construct_pyscf_system_ao
import numpy as np
from hartree_fock.mix import AlphaMixer, DIIS
import pytest


def test_rhf():

    r = 1.871
    molecule = f"O; H, 1, {r};  H 1 {r} 2 100.0"
    basis = "sto-3g"
    e_hf_pyscf = -74.965_900_173_175_2

    system = construct_pyscf_system_ao(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    rhf_diis = RHF(system, mixer=DIIS, verbose=False)
    rhf_diis.compute_ground_state(tol=1e-10)
    assert abs(rhf_diis.total_energy - e_hf_pyscf) < 1e-9

    rhf_alpha = RHF(system, mixer=AlphaMixer, verbose=False)
    mixer_kwargs = dict(theta=0.2)
    rhf_alpha.compute_ground_state(tol=1e-10, **mixer_kwargs)
    assert abs(rhf_alpha.total_energy - e_hf_pyscf) < 1e-9


@pytest.mark.skip
def test_tdho_rhf():
    n = 2
    l = 12
    radius = 10
    num_grid_points = 401

    tdho = TwoDimensionalHarmonicOscillator(n, l, radius, num_grid_points)
    tdho.setup_system(add_spin=False, anti_symmetrize=False)

    rhf = RHF(tdho, verbose=True)

    rhf.compute_ground_state(tol=1e-10)

    assert abs(rhf.compute_energy() - 3.162_691) < 1e-6


if __name__ == "__main__":
    test_rhf()

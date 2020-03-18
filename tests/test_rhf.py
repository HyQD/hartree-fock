from hartree_fock.rhf import RHF
from quantum_systems import TwoDimensionalHarmonicOscillator
from quantum_systems import construct_pyscf_system_ao
import numpy as np


def test_rhf():

    r = 1.871
    molecule = f"O; H, 1, {r};  H 1 {r} 2 100.0"
    basis = "sto-3g"

    system = construct_pyscf_system_ao(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    rhf = RHF(system, verbose=True)
    rhf.compute_ground_state(tol=1e-10)
    e_hf_pyscf = -74.965_900_173_175_2
    assert abs(rhf.total_energy - e_hf_pyscf) < 1e-10


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

from hartree_fock.ghf import GHF
from quantum_systems import TwoDimensionalHarmonicOscillator
from quantum_systems import construct_pyscf_system_ao
import numpy as np
import pytest


def test_h2o_ghf():

    r = 1.871
    molecule = f"O; H, 1, {r};  H 1 {r} 2 100.0"
    basis = "sto-3g"

    system = construct_pyscf_system_ao(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=True,
        anti_symmetrize=True,
    )

    ghf = GHF(system, verbose=True)
    ghf.compute_ground_state(tol=1e-12)
    e_hf_pyscf = -74.965_900_173_175_2

    assert abs(ghf.total_energy - e_hf_pyscf) < 1e-10


if __name__ == "__main__":
    test_ghf()

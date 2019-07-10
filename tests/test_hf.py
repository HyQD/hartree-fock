import numpy as np

from hartree_fock import HartreeFock
from quantum_systems import TwoDimensionalHarmonicOscillator


def test_tdho_hf():
    n = 2
    l = 12
    radius = 10
    num_grid_points = 401

    tdho = TwoDimensionalHarmonicOscillator(n, l, radius, num_grid_points)
    tdho.setup_system()

    hf = HartreeFock(tdho, verbose=True)

    hf.compute_ground_state(tol=1e-10)

    assert abs(hf.compute_energy() - 3.162691) < 1e-6

    rho_qp = hf.compute_one_body_density_matrix()

    np.testing.assert_allclose(rho_qp[tdho.o, tdho.o], np.identity(n))
    np.testing.assert_allclose(rho_qp[tdho.v, tdho.v], np.zeros((l - n, l - n)))

    rho = hf.compute_particle_density()

from hartree_fock import UHF
from quantum_systems import TwoDimensionalHarmonicOscillator


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

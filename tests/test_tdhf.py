import os
import numpy as np

from hartree_fock import TDHF
from hartree_fock.integrators import GaussIntegrator
from quantum_systems import ODQD
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import HOPotential
from quantum_systems.time_evolution_operators import LaserField


def test_tdhf():
    n = 2
    l = 20

    omega = 0.25
    grid_length = 10
    num_grid_points = 801
    a = 0.25
    alpha = 1

    laser_frequency = 8 * omega
    E = 1
    laser_pulse = lambda t: E * np.sin(laser_frequency * t)

    odho = ODQD(
        n,
        l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        a=a,
        alpha=alpha,
    )
    odho.setup_system(potential=HOPotential(omega=omega))
    odho.set_time_evolution_operator(LaserField(laser_pulse))

    integrator = GaussIntegrator(s=3, eps=1e-6, np=np)
    tdhf = TDHF(odho, integrator=integrator, verbose=True)
    tdhf.compute_ground_state(tol=1e-5)
    tdhf.set_initial_conditions()

    rho_tdhf = tdhf.compute_particle_density()
    test_rho = np.loadtxt(os.path.join("tests", "dat", "rho_tdhf_real.dat"))

    np.testing.assert_allclose(rho_tdhf.real, test_rho[:, 1], atol=1e-6)

    t_start = 0
    t_end = 4 * 2 * np.pi / laser_frequency
    dt = 1e-2

    num_timesteps = int((t_end - t_start) / dt + 1)
    time_points = np.linspace(t_start, t_end, num_timesteps)
    overlap = np.zeros(num_timesteps, dtype=np.complex128)
    overlap[0] = tdhf.compute_time_dependent_overlap()

    for i, amp in enumerate(tdhf.solve(time_points)):
        overlap[i + 1] = tdhf.compute_time_dependent_overlap()

    test_overlap = np.loadtxt(
        os.path.join("tests", "dat", "overlap_tdhf_real.dat")
    )

    np.testing.assert_allclose(overlap.real, test_overlap[:, 1], atol=1e-6)

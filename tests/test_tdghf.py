import os
import numpy as np

from scipy.integrate import complex_ode

from hartree_fock import TDGHF, GHF
from quantum_systems import ODQD, GeneralOrbitalSystem
from quantum_systems.time_evolution_operators import LaserField

from gauss_integrator import GaussIntegrator


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

    odho = GeneralOrbitalSystem(
        n,
        ODQD(
            l,
            grid_length=grid_length,
            num_grid_points=num_grid_points,
            a=a,
            alpha=alpha,
            potential=ODQD.HOPotential(omega=omega),
        ),
    )
    odho.set_time_evolution_operator(LaserField(laser_pulse))

    ghf = GHF(odho, verbose=True).compute_ground_state(tol=1e-5)
    assert abs(ghf.compute_energy() - 1.17959) < 1e-5

    tdghf = TDGHF(odho, verbose=True)
    r = complex_ode(tdghf).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(ghf.C.ravel())

    assert abs(tdghf.compute_energy(r.t, r.y) - ghf.compute_energy()) < 1e-7

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

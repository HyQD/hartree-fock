import os
import numpy as np
import pytest

from scipy.integrate import complex_ode

from hartree_fock import TDRHF, RHF
from quantum_systems import ODQD, GeneralOrbitalSystem
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from gauss_integrator import GaussIntegrator
from quantum_systems import construct_pyscf_system_ao, SpatialOrbitalSystem


def test_energy_conservation():
    molecule = "he 0.0 0.0 0.0"

    basis = "cc-pvdz"

    system = construct_pyscf_system_ao(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    rhf = RHF(system, verbose=False).compute_ground_state(
        tol=1e-12, change_system_basis=True
    )
    energy_gs = rhf.compute_energy()

    tdrhf = TDRHF(system, verbose=True)
    r = complex_ode(tdrhf).set_integrator("GaussIntegrator", s=3, eps=1e-10)
    r.set_initial_value(rhf.C.ravel())

    dt = 1e-2
    tfinal = 5
    num_steps = int(tfinal / dt) + 1
    time_points = np.linspace(0, tfinal, num_steps)

    energy = np.zeros(num_steps, dtype=np.complex128)

    i = 0

    while r.successful() and r.t < tfinal:
        assert abs(time_points[i] - r.t) < dt * 0.1

        energy[i] = tdrhf.compute_energy(r.t, r.y)

        i += 1

        r.integrate(time_points[i])

    energy[i] = tdrhf.compute_energy(r.t, r.y)

    assert np.linalg.norm(energy - energy_gs) < 1e-10


def test_helium():
    class sine_square_laser:
        def __init__(self, E0, omega, td, phase=0.0, start=0.0):
            self.F_str = E0
            self.omega = omega
            self.tprime = td
            self.phase = phase
            self.t0 = start

        def _phase(self, t):
            if callable(self.phase):
                return self.phase(t)
            else:
                return self.phase

        def __call__(self, t):
            dt = t - self.t0
            pulse = (
                (np.sin(np.pi * dt / self.tprime) ** 2)
                * np.heaviside(dt, 1.0)
                * np.heaviside(self.tprime - dt, 1.0)
                * np.sin(self.omega * dt + self._phase(dt))
                * self.F_str
            )
            return pulse

    molecule = "he 0.0 0.0 0.0"

    basis = "cc-pvdz"

    system = construct_pyscf_system_ao(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    rhf = RHF(system, verbose=False)
    rhf.compute_ground_state(tol=1e-12, change_system_basis=True)

    laser_pulse = sine_square_laser(E0=100.0, omega=2.87, td=5, phase=np.pi / 2)
    polarization = np.zeros(3)
    polarization_direction = 2
    polarization[polarization_direction] = 1
    system.set_time_evolution_operator(
        DipoleFieldInteraction(laser_pulse, polarization_vector=polarization)
    )

    tdrhf = TDRHF(system, verbose=True)
    r = complex_ode(tdrhf).set_integrator("GaussIntegrator", s=3, eps=1e-10)
    r.set_initial_value(rhf.C.ravel())

    dt = 1e-2
    tfinal = 5
    num_steps = int(tfinal / dt) + 1
    time_points = np.linspace(0, tfinal, num_steps)

    energy = np.zeros(num_steps, dtype=np.complex128)
    overlap = np.zeros(num_steps)
    dipole_moment = np.zeros((num_steps, 3), dtype=np.complex128)

    energy[0] = tdrhf.compute_energy(0, r.y.reshape(system.l, system.l))
    overlap[0] = tdrhf.compute_overlap(
        0, r.y.reshape(system.l, system.l), rhf.C
    )

    for j in range(3):
        dipole_moment[0, j] = -tdrhf.compute_one_body_expectation_value(
            0, r.y.reshape(system.l, system.l), system.position[j]
        )

    for i in range(num_steps - 1):

        r.integrate(r.t + dt)

        energy[i + 1] = tdrhf.compute_energy(
            r.t + dt, r.y.reshape(system.l, system.l)
        )

        overlap[i + 1] = tdrhf.compute_overlap(
            r.t + dt, r.y.reshape(system.l, system.l), rhf.C
        )

        for j in range(3):
            dipole_moment[i + 1, j] = -tdrhf.compute_one_body_expectation_value(
                r.t + dt, r.y.reshape(system.l, system.l), system.position[j]
            )

    test_overlap = np.load(
        os.path.join("tests", "dat", "tdrhf_helium_overlap.npy")
    )
    test_dip_z = np.load(os.path.join("tests", "dat", "tdrhf_helium_dip_z.npy"))
    test_energy = np.load(
        os.path.join("tests", "dat", "tdrhf_helium_energy.npy")
    )

    np.testing.assert_allclose(energy.real, test_energy.real, atol=1e-5)
    np.testing.assert_allclose(overlap, test_overlap, atol=1e-6)
    np.testing.assert_allclose(
        dipole_moment[:, 2].real, test_dip_z.real, atol=1e-6
    )

    tfinal_2 = 10
    time_points = np.linspace(tfinal, tfinal_2, num_steps)
    energy = np.zeros(num_steps, dtype=np.complex128)

    i = 0

    while r.successful() and r.t < tfinal_2:
        assert abs(time_points[i] - r.t) < dt * 0.1

        energy[i] = tdrhf.compute_energy(r.t, r.y)

        i += 1

        r.integrate(time_points[i])

    energy[i] = tdrhf.compute_energy(r.t, r.y)

    assert np.linalg.norm(energy - energy[0]) < 1e-10


def test_tdrhf():
    n = 2
    l = 10

    omega = 0.25
    grid_length = 10
    num_grid_points = 801
    a = 0.25
    alpha = 1

    laser_frequency = 8 * omega
    E = 1
    laser_pulse = lambda t: E * np.sin(laser_frequency * t)
    polarization = np.zeros(1)
    polarization[0] = 1

    odho = SpatialOrbitalSystem(
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
    odho.set_time_evolution_operator(
        DipoleFieldInteraction(laser_pulse, polarization_vector=polarization)
    )

    rhf = RHF(odho, verbose=True).compute_ground_state(tol=1e-7)
    assert abs(rhf.compute_energy() - 1.17959) < 1e-4

    tdrhf = TDRHF(odho, verbose=True)
    r = complex_ode(tdrhf).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(rhf.C.ravel())

    c_0 = r.y.copy()

    assert abs(tdrhf.compute_energy(r.t, r.y) - rhf.compute_energy()) < 1e-7

    rho_tdrhf = tdrhf.compute_particle_density(r.t, r.y)
    test_rho = np.loadtxt(os.path.join("tests", "dat", "rho_tdhf_real.dat"))

    np.testing.assert_allclose(rho_tdrhf.real, test_rho[:, 1], atol=1e-3)

    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(odho._basis_set.grid, rho_tdrhf.real, label="TDRF")
    # plt.plot(test_rho[:, 0], test_rho[:, 1], label="Test")
    # plt.grid()
    # plt.legend()
    # plt.show()
    # wat

    t_start = 0
    t_end = 4 * 2 * np.pi / laser_frequency
    dt = 1e-2

    num_timesteps = int((t_end - t_start) / dt + 1)
    time_points = np.linspace(t_start, t_end, num_timesteps)

    overlap = np.zeros(num_timesteps, dtype=np.complex128)

    i = 0

    while r.successful() and r.t < t_end:
        assert abs(time_points[i] - r.t) < dt * 0.1

        overlap[i] = tdrhf.compute_overlap(r.t, r.y, c_0)

        i += 1
        r.integrate(time_points[i])

    overlap[i] = tdrhf.compute_overlap(r.t, r.y, c_0)

    test_overlap = np.loadtxt(
        os.path.join("tests", "dat", "overlap_tdhf_real.dat")
    )

    np.testing.assert_allclose(overlap.real, test_overlap[:, 1], atol=1e-3)

    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(time_points, overlap, label="TDRHF")
    # plt.plot(time_points, test_overlap[:, 1].real, label="Test")
    # plt.grid()
    # plt.legend()

    # plt.show()

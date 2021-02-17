import os
import numpy as np
import pytest

from scipy.integrate import complex_ode

from hartree_fock import TDGHF, GHF
from quantum_systems import ODQD, GeneralOrbitalSystem
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from gauss_integrator import GaussIntegrator
from quantum_systems import construct_pyscf_system_ao

from matplotlib import pyplot as plt


@pytest.mark.skip
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
        add_spin=True,
        anti_symmetrize=True,
    )

    ghf = GHF(system, verbose=False)
    ghf.compute_ground_state(tol=1e-12, change_system_basis=True)

    laser_pulse = sine_square_laser(E0=100.0, omega=2.87, td=5, phase=np.pi / 2)
    polarization = np.zeros(3)
    polarization_direction = 2
    polarization[polarization_direction] = 1
    system.set_time_evolution_operator(
        DipoleFieldInteraction(laser_pulse, polarization_vector=polarization)
    )

    tdghf = TDGHF(system, verbose=True)
    r = complex_ode(tdghf).set_integrator("GaussIntegrator", s=3, eps=1e-10)
    r.set_initial_value(ghf.C.ravel())

    dt = 1e-2
    tfinal = 5
    num_steps = int(tfinal / dt) + 1
    print(f"num_steps={num_steps}")
    time_points = np.linspace(0, tfinal, num_steps)

    energy = np.zeros(num_steps, dtype=np.complex128)
    overlap = np.zeros(num_steps)
    dipole_moment = np.zeros((num_steps, 3), dtype=np.complex128)

    energy[0] = tdghf.compute_energy(0, r.y.reshape(system.l, system.l))
    overlap[0] = tdghf.compute_overlap(
        0, r.y.reshape(system.l, system.l), ghf.C
    )

    for j in range(3):
        dipole_moment[0, j] = -tdghf.compute_one_body_expectation_value(
            0, r.y.reshape(system.l, system.l), system.position[j]
        )

    for i in range(num_steps - 1):

        r.integrate(r.t + dt)

        energy[i + 1] = tdghf.compute_energy(
            r.t + dt, r.y.reshape(system.l, system.l)
        )

        overlap[i + 1] = tdghf.compute_overlap(
            r.t + dt, r.y.reshape(system.l, system.l), ghf.C
        )

        for j in range(3):
            dipole_moment[i + 1, j] = -tdghf.compute_one_body_expectation_value(
                r.t + dt, r.y.reshape(system.l, system.l), system.position[j]
            )

        print(i)

    plt.figure()
    plt.plot(time_points, energy.real)

    plt.figure()
    plt.plot(time_points, overlap)

    plt.figure()
    plt.plot(time_points, dipole_moment[:, polarization_direction].real)

    plt.show()


@pytest.mark.skip
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
    polarization = np.zeros(1)
    polarization[0] = 1

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
    odho.set_time_evolution_operator(
        DipoleFieldInteraction(laser_pulse, polarization_vector=polarization)
    )

    ghf = GHF(odho, verbose=True).compute_ground_state(tol=1e-5)
    assert abs(ghf.compute_energy() - 1.17959) < 1e-5

    tdghf = TDGHF(odho, verbose=True)
    r = complex_ode(tdghf).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(ghf.C.ravel())

    assert abs(tdghf.compute_energy(r.t, r.y) - ghf.compute_energy()) < 1e-7

    # rho_tdhf = tdghf.compute_particle_density()
    # test_rho = np.loadtxt(os.path.join("tests", "dat", "rho_tdhf_real.dat"))

    # np.testing.assert_allclose(rho_tdhf.real, test_rho[:, 1], atol=1e-6)

    t_start = 0
    t_end = 4 * 2 * np.pi / laser_frequency
    dt = 1e-2

    num_timesteps = int((t_end - t_start) / dt + 1)
    time_points = np.linspace(t_start, t_end, num_timesteps)
    overlap = np.zeros(num_timesteps, dtype=np.complex128)
    overlap[0] = tdghf.compute_overlap(0, r.y.reshape(odho.l, odho.l), ghf.C)

    for i in range(num_timesteps - 1):

        r.integrate(r.t + dt)

        overlap[i + 1] = tdghf.compute_overlap(
            r.t + dt, r.y.reshape(odho.l, odho.l), ghf.C
        )

        print(i)

    test_overlap = np.loadtxt(
        os.path.join("tests", "dat", "overlap_tdhf_real.dat")
    )

    # np.testing.assert_allclose(overlap.real, test_overlap[:, 1], atol=1e-6)
    plt.figure()
    plt.plot(time_points, overlap)
    plt.plot(time_points, test_overlap[:, 1].real)

    plt.show()


if __name__ == "__main__":
    test_tdhf()

import os
import numpy as np
import pytest

from scipy.integrate import complex_ode

from hartree_fock import TDRHF, RHF
from quantum_systems import ODQD, GeneralOrbitalSystem
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from gauss_integrator import GaussIntegrator
from quantum_systems import construct_pyscf_system_ao


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

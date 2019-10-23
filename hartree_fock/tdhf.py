import warnings

from hartree_fock import HartreeFock
from hartree_fock.integrators import RungeKutta4
from hartree_fock.hf_helper import (
    create_orthogonalizer,
    build_density_matrix,
    build_general_fock_matrix,
    compute_general_hf_energy,
    compute_particle_density,
)


class TDHF:
    hf_class = HartreeFock

    def __init__(
        self, system, np=None, integrator=None, td_verbose=False, **hf_kwargs
    ):
        if np is None:
            import numpy as np

        self.np = np

        if not "np" in hf_kwargs:
            hf_kwargs["np"] = self.np

        self.verbose = td_verbose
        self.system = system

        self.h = self.system.h
        self.u = self.system.u
        self.o = self.system.o

        self.hf = self.hf_class(self.system, **hf_kwargs)

        if integrator is None:
            integrator = RungeKutta4(np=self.np)

        self.integrator = integrator.set_rhs(self)
        self._C = None

        self.compute_ground_state_energy = self.hf.compute_energy
        self.compute_ground_state_particle_density = (
            self.hf.compute_particle_density
        )
        self.compute_ground_state_one_body_desnity_matrix = (
            self.hf.compute_one_body_density_matrix
        )

    def compute_ground_state(self, *args, **kwargs):
        if "change_system_basis" not in kwargs:
            kwargs["change_system_basis"] = True

        self.hf.compute_ground_state(*args, **kwargs)

        # Make sure that we have an orthonormal basis
        if not self.np.allclose(self.system.s, self.np.identity(self.system.l)):
            warnings.warn(
                f"Ground state basis is not orthonormal. "
                + f"The {self.__class__.name} assumes an orthonormal "
                + f"spin-orbital basis"
            )

    def set_initial_conditions(self, C=None):
        if C is None:
            C = self.np.eye(self.system.l)

        self._C_0 = C.copy()
        self._C = C

    def solve(self, time_points, timestep_tol=1e-8):
        n = len(time_points)

        for i in range(n - 1):
            dt = time_points[i + 1] - time_points[i]
            C_t = self.integrator.step(self._C.ravel(), time_points[i], dt)
            self._C = C_t.reshape(self._C.shape)

            if abs(self.last_timestep - (time_points[i] + dt)) > timestep_tol:
                self.last_timestep = time_points[i] + dt
                self.update_hamiltonian(self.last_timestep)

            yield self._C

    def update_hamiltonian(self, current_time):
        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)

    @property
    def C(self):
        return self._C

    def compute_one_body_density_matrix(self):
        return build_density_matrix(self._C, self.o, self.np)

    def compute_particle_density(self):
        rho_qp = self.compute_one_body_density_matrix()

        return compute_particle_density(rho_qp, self.system.spf, self.np)

    def compute_time_dependent_overlap(self):
        np = self.np

        S = self._C[:, self.o].conj().T @ self._C_0[:, self.o]

        return np.abs(np.linalg.det(S)) ** 2

    def compute_energy(self):
        density_matrix = build_density_matrix(self._C, self.o, self.np)

        return compute_general_hf_energy(
            density_matrix,
            self.h,
            self.u,
            self.system.nuclear_repulsion_energy,
            self.np,
        )

    def __call__(self, C, current_time):
        C = C.reshape(self._C.shape)

        density_matrix = build_density_matrix(C, self.o, self.np)
        self.update_hamiltonian(current_time)
        fock_matrix = build_general_fock_matrix(
            self.h, self.u, density_matrix, self.np
        )

        self.last_timestep = current_time

        return -1j * self.np.dot(fock_matrix, C).ravel()

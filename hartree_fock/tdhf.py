import abc
import warnings


class TimeDependentHartreeFock(metaclass=abc.ABCMeta):
    def __init__(self, system, verbose=False):
        self.verbose = verbose

        self.system = system

        self.h = self.system.h
        self.u = self.system.u
        self.fock_matrix = self.system.construct_fock_matrix(self.h, self.u)

        self.np = self.system.np

        self.last_timestep = None

    @abc.abstractmethod
    def compute_energy(self, current_time, C):
        pass

    def compute_one_body_expectation_value(self, current_time, C, mat):
        D = self.compute_one_body_density_matrix(current_time, C)
        return self.np.trace(self.np.dot(D, mat))

    def compute_two_body_expectation_value(self, current_time, C, op):
        rho_rspq = self.compute_two_body_density_matrix(current_time, C)

        return 0.5 * self.np.einsum("pqrs, rspq ->", op, rho_rspq)

    @abc.abstractmethod
    def compute_one_body_density_matrix(self, current_time, C):
        pass

    @abc.abstractmethod
    def compute_two_body_density_matrix(self):
        pass

    def compute_particle_density(self, current_time, C):

        np = self.np

        rho_qp = self.compute_one_body_density_matrix(current_time, C)

        if np.abs(np.trace(rho_qp) - self.system.n) > 1e-8:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(np.trace(rho_qp), self.system.n)
            warnings.warn(warn)

        return self.system.compute_particle_density(rho_qp)

    @abc.abstractmethod
    def compute_overlap(self, current_time, C_a, C_b):
        pass

    @abc.abstractmethod
    def build_density_matrix(self, C):
        pass

    @abc.abstractmethod
    def build_fock_matrix(self, h, u, density_matrix):
        pass

    def update_hamiltonian(self, current_time):
        if current_time == self.last_timestep:
            return

        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)

        self.last_timestep = current_time

    def __call__(self, current_time, prev_C):
        C = prev_C.reshape(self.system.l, self.system.l)

        self.update_hamiltonian(current_time)

        density_matrix = self.build_density_matrix(C)

        self.fock_matrix = self.build_fock_matrix(
            self.h, self.u, density_matrix
        )

        self.last_timestep = current_time

        return -1j * self.np.dot(self.fock_matrix, C).ravel()


# from hartree_fock import GHF
# from hartree_fock.integrators import RungeKutta4
# from hartree_fock.hf_helper import (
#     create_orthogonalizer,
#     build_density_matrix,
#     build_general_fock_matrix,
#     compute_general_hf_energy,
# )
#
#
# class TDHF:
#     hf_class = GHF
#
#     def __init__(
#         self, system, np=None, integrator=None, td_verbose=False, **hf_kwargs
#     ):
#         if np is None:
#             import numpy as np
#
#         self.np = np
#
#         if not "np" in hf_kwargs:
#             hf_kwargs["np"] = self.np
#
#         self.verbose = td_verbose
#         self.system = system
#
#         self.h = self.system.h
#         self.u = self.system.u
#         self.o = self.system.o
#
#         self.hf = self.hf_class(self.system, **hf_kwargs)
#
#         if integrator is None:
#             integrator = RungeKutta4(np=self.np)
#
#         self.integrator = integrator.set_rhs(self)
#         self._C = None
#
#         self.compute_ground_state_energy = self.hf.compute_energy
#         self.compute_ground_state_particle_density = (
#             self.hf.compute_particle_density
#         )
#         self.compute_ground_state_one_body_desnity_matrix = (
#             self.hf.compute_one_body_density_matrix
#         )
#
#     def compute_ground_state(self, *args, **kwargs):
#         if "change_system_basis" not in kwargs:
#             kwargs["change_system_basis"] = True
#
#         self.hf.compute_ground_state(*args, **kwargs)
#
#         # Make sure that we have an orthonormal basis
#         if not self.np.allclose(self.system.s, self.np.identity(self.system.l)):
#             warnings.warn(
#                 f"Ground state basis is not orthonormal. "
#                 + f"The {self.__class__.name} assumes an orthonormal "
#                 + f"spin-orbital basis"
#             )
#
#     def set_initial_conditions(self, C=None):
#         if C is None:
#             C = self.np.eye(self.system.l)
#
#         C = C.astype(self.np.complex128)
#         self._C_0 = C.copy()
#         self._C = C
#
#     def solve(self, time_points, timestep_tol=1e-8):
#         n = len(time_points)
#
#         for i in range(n - 1):
#             dt = time_points[i + 1] - time_points[i]
#             C_t = self.integrator.step(self._C.ravel(), time_points[i], dt)
#             self._C = C_t.reshape(self._C.shape)
#
#             if abs(self.last_timestep - (time_points[i] + dt)) > timestep_tol:
#                 self.last_timestep = time_points[i] + dt
#                 self.update_hamiltonian(self.last_timestep)
#
#             yield self._C
#
#     def update_hamiltonian(self, current_time):
#         self.h = self.system.h_t(current_time)
#         self.u = self.system.u_t(current_time)
#
#     @property
#     def C(self):
#         return self._C
#
#     def compute_one_body_density_matrix(self):
#         return build_density_matrix(self._C, self.o, self.np)
#
#     def compute_particle_density(self):
#         rho_qp = self.compute_one_body_density_matrix()
#
#         return self.system.compute_particle_density(rho_qp, self._C)
#
#     def compute_time_dependent_overlap(self):
#         np = self.np
#
#         S = self._C[:, self.o].conj().T @ self._C_0[:, self.o]
#
#         return np.abs(np.linalg.det(S)) ** 2
#
#     def compute_energy(self):
#         density_matrix = build_density_matrix(self._C, self.o, self.np)
#
#         return compute_general_hf_energy(
#             density_matrix,
#             self.h,
#             self.u,
#             self.system.nuclear_repulsion_energy,
#             self.np,
#         )
#
#     def __call__(self, C, current_time):
#         C = C.reshape(self._C.shape)
#
#         density_matrix = build_density_matrix(C, self.o, self.np)
#         self.update_hamiltonian(current_time)
#         fock_matrix = build_general_fock_matrix(
#             self.h, self.u, density_matrix, self.np
#         )
#
#         self.last_timestep = current_time
#
#         return -1j * self.np.dot(fock_matrix, C).ravel()

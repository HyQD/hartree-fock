import scipy.linalg

from hartree_fock.hf_helper import (
    build_density_matrix,
    build_general_fock_matrix,
    compute_error_vector,
    compute_particle_density,
    compute_general_hf_energy,
)
from hartree_fock.mix import EmptyMixer


class HartreeFock:
    def __init__(self, system, mixer=EmptyMixer, verbose=False, np=None):
        if np is None:
            import numpy as np

        self.np = np
        self.system = system
        self.mixer = mixer
        self.verbose = verbose

        self.o = self.system.o

    def compute_initial_guess(self):
        # compute initial guess from the one-body part of the hamiltonian and
        # the overlap.
        self._epsilon, self._C = self.diagonalize(self.system.h, self.system.s)
        self.density_matrix = self.build_density_matrix()
        self.fock_matrix = self.build_fock_matrix()

    def compute_energy(self):
        return compute_general_hf_energy(
            self.density_matrix,
            self.system.h,
            self.system.u,
            self.system.nuclear_repulsion_energy,
            self.np,
        )

    def compute_one_body_density_matrix(self):
        return build_density_matrix(self._C, self.o, self.np)

    def compute_particle_density(self):
        np = self.np

        rho_qp = self.compute_one_body_density_matrix()
        spf = np.tensordot(self.C, self.system.spf, axes=((0), (0)))

        return compute_particle_density(rho_qp, spf, self.np)

    def build_fock_matrix(self):
        return build_general_fock_matrix(
            self.system.h, self.system.u, self.density_matrix, self.np
        )

    def build_error_vector(self):
        return compute_error_vector(
            self.fock_matrix, self.density_matrix, self.system.s
        )

    def build_density_matrix(self):
        return build_density_matrix(self._C, self.o, self.np)

    def setup_mixer(self, **mixer_kwargs):
        self.fock_mixer = self.mixer(**mixer_kwargs)

    def diagonalize(self, fock_matrix, overlap):
        return scipy.linalg.eigh(fock_matrix, overlap)

    def compute_scf_iteration(self):
        # Solve the Roothan-Hall equations
        self._epsilon, self._C = self.diagonalize(
            self.fock_matrix, self.system.s
        )
        self.density_matrix = self.build_density_matrix()

        trial_vector = self.fock_matrix
        direction_vector = self.build_fock_matrix()
        error_vector = self.build_error_vector()

        self.compute_new_fock_matrix(
            trial_vector, direction_vector, error_vector
        )

    def compute_new_fock_matrix(
        self, trial_vector, direction_vector, error_vector
    ):
        self.fock_matrix = self.fock_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_density_residual(self, prev_density_matrix, density_matrix):
        if not type(prev_density_matrix) in [list, tuple, set]:
            prev_density_matrix = [prev_density_matrix]
            density_matrix = [density_matrix]

        return [
            self.np.linalg.norm(d_prev - d)
            for d_prev, d in zip(prev_density_matrix, density_matrix)
        ]

    def compute_ground_state(
        self,
        max_iterations=100,
        tol=1e-4,
        change_system_basis=False,
        **mixer_kwargs,
    ):
        if not "np" in mixer_kwargs:
            mixer_kwargs["np"] = self.np

        self.setup_mixer(**mixer_kwargs)
        self.compute_initial_guess()

        density_residual = [100]

        for i in range(max_iterations):
            if self.verbose:
                print(
                    f"{self.__class__.__name__} energy: "
                    + f"{self.compute_energy()} @ iteration: {i}\t"
                    + f"residual: {density_residual}"
                )

            if all(d_residual < tol for d_residual in density_residual):
                break

            self.prev_density_matrix = self.density_matrix
            self.compute_scf_iteration()

            density_residual = self.compute_density_residual(
                self.prev_density_matrix, self.density_matrix
            )

        self._epsilon, self._C = self.diagonalize(
            self.fock_matrix, self.system.s
        )
        self.density_matrix = self.build_density_matrix()

        if self.verbose:
            print(
                f"Final {self.__class__.__name__} energy: "
                + f"{self.compute_energy()} @ iteration: {i}\t"
                + f"residual: {density_residual}"
            )

        assert (
            i < max_iterations - 1
        ), f"{self.__class__.__name__} solver did not converge."

        if change_system_basis:
            self.change_basis()

    def change_basis(self):
        if self.verbose:
            print(f"Changing to {self.__class__.__name__} basis")

        self.system.change_basis(self._C)
        self._C = self.np.identity(self.system.l)
        self.density_matrix = self.build_density_matrix()
        self.fock_matrix = self.build_fock_matrix()

    @property
    def C(self):
        return self._C

    @property
    def epsilon(self):
        return self._epsilon

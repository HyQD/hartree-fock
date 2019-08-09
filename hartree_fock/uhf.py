import scipy.linalg

from hartree_fock import HartreeFock
from hartree_fock.hf_helper import (
    build_density_matrix,
    build_uhf_fock_matrices,
    compute_error_vector,
)


class UHF(HartreeFock):
    def compute_energy(self):
        np = self.np

        D_up, D_down = self.density_matrix
        D_sum = D_up + D_down

        energy = np.trace(np.dot(D_sum, self.system.h))
        energy += 0.5 * np.trace(
            np.dot(
                D_sum, np.tensordot(D_sum, self.system.u, axes=((0, 1), (3, 1)))
            )
        )
        energy -= 0.5 * np.trace(
            np.dot(
                D_up, np.tensordot(D_up, self.system.u, axes=((0, 1), (2, 1)))
            )
        )
        energy -= 0.5 * np.trace(
            np.dot(
                D_down,
                np.tensordot(D_down, self.system.u, axes=((0, 1), (2, 1))),
            )
        )

        return energy + self.system.nuclear_repulsion_energy

    def compute_initial_guess(self):
        """Bleh
        """
        # compute initial guess from the one-body part of the hamiltonian and
        # the overlap.
        self._epsilon, self._C = self.diagonalize(
            (self.system.h, self.system.h), self.system.s
        )
        self.density_matrix = self.build_density_matrix()
        self.fock_matrix = self.build_fock_matrix()

    def diagonalize(self, fock_matrix, overlap):
        epsilon_up, C_up = scipy.linalg.eigh(fock_matrix[0], overlap)
        epsilon_down, C_down = scipy.linalg.eigh(fock_matrix[1], overlap)

        return (epsilon_up, epsilon_down), (C_up, C_down)

    def build_density_matrix(self):
        C_up, C_down = self._C

        D_up = build_density_matrix(C_up, self.system.o_up, self.np)
        D_down = build_density_matrix(C_down, self.system.o_down, self.np)

        return (D_up, D_down)

    def build_fock_matrix(self):
        return build_uhf_fock_matrices(
            self.system.h, self.system.u, self.density_matrix, self.np
        )

    def setup_mixer(self, **mixer_kwargs):
        self.fock_mixer_up = self.mixer(**mixer_kwargs)
        self.fock_mixer_down = self.mixer(**mixer_kwargs)

    def build_error_vector(self):
        error_up = compute_error_vector(
            self.fock_matrix[0], self.density_matrix[0], self.system.s
        )
        error_down = compute_error_vector(
            self.fock_matrix[1], self.density_matrix[1], self.system.s
        )

        return error_up, error_down

    def compute_new_fock_matrix(
        self, trial_vector, direction_vector, error_vector
    ):
        F_up = self.fock_mixer_up.compute_new_vector(
            trial_vector[0], direction_vector[0], error_vector[0]
        )
        F_down = self.fock_mixer_down.compute_new_vector(
            trial_vector[1], direction_vector[1], error_vector[1]
        )

        self.fock_matrix = (F_up, F_down)

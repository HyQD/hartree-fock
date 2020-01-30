import scipy.linalg

from hartree_fock import GHF
from hartree_fock.hf_helper import (
    build_density_matrix,
    build_uhf_fock_matrices,
    compute_error_vector,
)


class UHF(GHF):
    spin_directions = {
        "up": 0,
        "down": 1,
        "alpha": 0,
        "beta": 1,
        "a": 0,
        "b": 1,
    }

    def compute_energy(self):
        np = self.np

        D_a, D_b = self.density_matrix
        D_sum = D_a + D_b

        energy = np.trace(np.dot(D_sum, self.system.h))
        energy += 0.5 * np.trace(
            np.dot(
                D_sum, np.tensordot(D_sum, self.system.u, axes=((0, 1), (3, 1)))
            )
        )
        energy -= 0.5 * np.trace(
            np.dot(D_a, np.tensordot(D_a, self.system.u, axes=((0, 1), (2, 1))))
        )
        energy -= 0.5 * np.trace(
            np.dot(
                D_b, np.tensordot(D_b, self.system.u, axes=((0, 1), (2, 1))),
            )
        )

        return energy + self.system.nuclear_repulsion_energy

    def compute_one_body_density_matrix(self, direction="up"):
        ind = self.spin_directions[direction.lower()]

        o = self.system.o_a if ind == 0 else self.system.o_b

        rho_qp = self.np.zeros_like(self.system.h)
        rho_qp[o, o] = (
            self._C[ind][:, o].conj().T @ self.system.s @ self._C[ind][:, o]
        )

        return rho_qp

    def compute_particle_density(self, direction="up"):
        ind = self.spin_directions[direction.lower()]

        rho_qp = self.compute_one_body_density_matrix(direction=direction)

        return self.system.compute_particle_density(rho_qp, c=self._C[ind])

    def compute_initial_guess(self):
        """Function computing the first iteration of the SCF-procedure using
        the UHF ansatz.
        """
        # compute initial guess from the one-body part of the hamiltonian and
        # the overlap.
        self._epsilon, self._C = self.diagonalize(
            (self.system.h, self.system.h), self.system.s
        )
        self.density_matrix = self.build_density_matrix()
        self.fock_matrix = self.build_fock_matrix()

    def diagonalize(self, fock_matrix, overlap):
        epsilon_a, C_a = scipy.linalg.eigh(fock_matrix[0], overlap)
        epsilon_b, C_b = scipy.linalg.eigh(fock_matrix[1], overlap)

        return (epsilon_a, epsilon_b), (C_a, C_b)

    def build_density_matrix(self):
        C_a, C_b = self._C

        D_a = build_density_matrix(C_a, self.system.o_a, self.np)
        D_b = build_density_matrix(C_b, self.system.o_b, self.np)

        return (D_a, D_b)

    def build_fock_matrix(self):
        return build_uhf_fock_matrices(
            self.system.h, self.system.u, self.density_matrix, self.np
        )

    def setup_mixer(self, **mixer_kwargs):
        self.fock_mixer_a = self.mixer(**mixer_kwargs)
        self.fock_mixer_b = self.mixer(**mixer_kwargs)

    def build_error_vector(self):
        error_a = compute_error_vector(
            self.fock_matrix[0], self.density_matrix[0], self.system.s
        )
        error_b = compute_error_vector(
            self.fock_matrix[1], self.density_matrix[1], self.system.s
        )

        return error_a, error_b

    def compute_new_fock_matrix(
        self, trial_vector, direction_vector, error_vector
    ):
        F_a = self.fock_mixer_a.compute_new_vector(
            trial_vector[0], direction_vector[0], error_vector[0]
        )
        F_b = self.fock_mixer_b.compute_new_vector(
            trial_vector[1], direction_vector[1], error_vector[1]
        )

        self.fock_matrix = (F_a, F_b)

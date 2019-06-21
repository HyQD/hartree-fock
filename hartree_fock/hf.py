import scipy.linalg

from hartree_fock.hf_helper import (
    build_density_matrix,
    build_general_fock_matrix,
    compute_error_vector,
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

        self.h = self.system.h
        self.u = self.system.u
        self.s = self.system.s

        self.o = self.system.o

    def compute_initial_guess(self):
        # Compute initial guess from the one-body part of the Hamiltonian and
        # the overlap.
        self._epsilon, self._C = scipy.linalg.eigh(self.h, self.s)
        self.density_matrix = self.build_density_matrix()
        self.fock_matrix = self.build_fock_matrix()

    def compute_energy(self):
        np = self.np

        # energy <- D_{ba} h_{ab}
        energy = np.trace(np.dot(self.density_matrix, self.h))

        # term_{bd} = 0.5 * D_{ca} u^{ab}_{cd}
        term = 0.5 * np.tensordot(
            self.density_matrix, self.u, axes=((0, 1), (2, 0))
        )
        # energy <- D_{db} term_{bd}
        energy += np.trace(np.dot(self.density_matrix, term))

        return energy

    def build_fock_matrix(self):
        return build_general_fock_matrix(
            self.h, self.u, self.density_matrix, self.np
        )

    def build_error_vector(self):
        return compute_error_vector(
            self.fock_matrix, self.density_matrix, self.s
        )

    def build_density_matrix(self):
        return build_density_matrix(self._C, self.o, self.np)

    def setup_mixer(self, **mixer_kwargs):
        self.fock_mixer = self.mixer(**mixer_kwargs)

    def compute_scf_iteration(self):
        # Solve the Roothan-Hall equations
        self._epsilon, self._C = scipy.linalg.eigh(self.fock_matrix, self.s)
        self.density_matrix = self.build_density_matrix()

        trial_vector = self.fock_matrix
        direction_vector = self.build_fock_matrix()
        error_vector = self.build_error_vector()

        self.fock_matrix = self.fock_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_ground_state(
        self, max_iterations=100, tol=1e-4, **mixer_kwargs
    ):
        if not "np" in mixer_kwargs:
            mixer_kwargs["np"] = self.np

        self.setup_mixer(**mixer_kwargs)
        self.compute_initial_guess()
        energy = self.compute_energy()

        diff = 100

        for i in range(max_iterations):
            if self.verbose:
                print(
                    f"{self.__class__.__name__} energy: "
                    + f"{energy} @ iteration: {i}"
                )

            if diff < tol:
                break

            self.compute_scf_iteration()

            energy_prev = energy
            energy = self.compute_energy()

            diff = abs(energy - energy_prev)

    @property
    def C(self):
        return self._C

    @property
    def epsilon(self):
        return self._epsilon

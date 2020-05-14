import abc
import warnings
import scipy.linalg
from hartree_fock.mix import DIIS


class HartreeFock(metaclass=abc.ABCMeta):
    """Hartree Fock Abstract class

    Abstract base class defining the skeleton of a
    Hartree Fock ground state solver class.

    Parameters
    ----------
    system : QuantumSystems
        Quantum systems class instance
    mixer : AlphaMixer
        AlpaMixer object
    verbose : bool
        Prints iterations for ground state computation if True
    """

    def __init__(self, system, mixer=DIIS, verbose=False):
        self.np = system.np

        self.system = system
        self.verbose = verbose
        self.mixer = mixer

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m

        self.h = self.system.h
        self.u = self.system.u
        self._total_energy = 0
        self.f = self.system.construct_fock_matrix(self.h, self.u)
        self.initial_guess("identity")

        self.o, self.v = self.system.o, self.system.v

    def initial_guess(self, key):
        """
        Various initial guesses are used in the litterature.
        """
        # self._epsilon = self.np.diag(self.system.h)
        # self._C = self.np.eye(self.system.l)
        self._epsilon, self._C = self.diagonalize(self.system.h, self.system.s)
        self.density_matrix = self.build_density_matrix(self._C)
        self.f = self.build_fock_matrix(self.density_matrix)

    def density_matrix(self):
        return self.build_density_matrix(self._C)

    @abc.abstractmethod
    def build_density_matrix(self, C):
        pass

    @abc.abstractmethod
    def build_fock_matrix(self, P):
        pass

    @abc.abstractmethod
    def _compute_energy(self, P, F):
        pass

    def compute_energy(self):
        return self._compute_energy(self.density_matrix, self.f)

    def compute_ground_state(
        self, max_iterations=100, tol=1e-4, **mixer_kwargs
    ):

        converged = False
        energy_prev = 1e10

        if not "np" in mixer_kwargs:
            mixer_kwargs["np"] = self.np

        self.mixer = self.mixer(**mixer_kwargs)

        for i in range(1, max_iterations):

            self.f = self.build_fock_matrix(self.density_matrix)
            self._total_energy = self._compute_energy(
                self.density_matrix, self.f
            )

            converged = abs(self._total_energy - energy_prev) < tol
            if self.verbose:
                print(
                    f"converged={converged}, total_energy={self._total_energy}, iterations={i}"
                )
            if converged:
                break
            energy_prev = self._total_energy

            # At convergence f_{ia} = 0, when f is the transformed fock matrix (f in MO basis)
            error_vector = (self._C.conj().T @ self.f @ self._C)[
                self.o, self.v
            ]  # Eq. [10.6.23] in Helgaker et. al.
            # the current fock matrix is the trial_vector
            f_mixed = self.mixer.compute_new_vector(self.f, error_vector)
            self._epsilon, self._C = self.diagonalize(f_mixed, self.system.s)
            self.density_matrix = self.build_density_matrix(self._C)

    @staticmethod
    def diagonalize(A, S):
        """
        Solve the generalized eigenvalue problem AC = SCE,
        where E = diag(e1,...,eL)
        """
        return scipy.linalg.eigh(A, S)

    def change_basis(self):
        """Function changing the system basis to the Hartree-Fock basis."""
        if self.verbose:
            print(f"Changing to {self.__class__.__name__} basis")

        self.system.change_basis(self._C)
        self._C = self.np.identity(len(self._C))
        self.density_matrix = self.build_density_matrix(self._C)
        self.fock_matrix = self.build_fock_matrix(self.density_matrix)

    @property
    def C(self):
        return self._C

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def total_energy(self):
        return self._total_energy

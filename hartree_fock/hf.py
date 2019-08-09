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
    """Class implementing a Hartree-Fock solver using general spin-orbitals.

    Parameters
    ----------
    system : QuantumSystem
        System containing matrix elements needed by the solver.
    mixer : Mixer
        Convergence mixer for the self-consistent field procedure. Default is
        ``EmptyMixer`` which does no mixing. This is enough in many cases.
    verbose : bool
        Variable to toggle the output of convergence info and converged energy.
        Default is ``False``, i.e., quiet-mode.
    np : module
        Matrix library, default is ``None`` which imports ``numpy``.
    """

    def __init__(self, system, mixer=EmptyMixer, verbose=False, np=None):
        if np is None:
            import numpy as np

        self.np = np
        self.system = system
        self.mixer = mixer
        self.verbose = verbose

        self.o = self.system.o

    def compute_initial_guess(self):
        r"""Compute initial guess for the self-consistent field procedure using
        the one-body Hamiltonian as the initial Fock matrix and solving the
        Roothan-Hall equations.
        That is, the function solves the generalized eigenvalue equation

        .. math:: \boldsymbol{h}\boldsymbol{C}
            = \boldsymbol{S}\boldsymbol{C}\boldsymbol{\epsilon},

        where :math:`\boldsymbol{h}` is the one-body Hamiltonian,
        :math:`\boldsymbol{C}` the coefficient matrix as the eigenvectors,
        :math:`\boldsymbol{S}` the overlap matrix from the system, and
        :math:`\boldsymbol{\epsilon} = \text{diag}(\epsilon_1, \dots)` the
        diagonal matrix with the eigenvalues from the one-body Hamiltonian.
        The function proceeds by building and storing the density matrix
        :math:`\boldsymbol{D}` from the coefficient matrix, and the initial
        Fock matrix :math:`\boldsymbol{F}`.

        Note that this function is called internally by the
        ``compute_ground_state``-function.
        """
        # compute initial guess from the one-body part of the hamiltonian and
        # the overlap.
        self._epsilon, self._C = self.diagonalize(self.system.h, self.system.s)
        self.density_matrix = self.build_density_matrix()
        self.fock_matrix = self.build_fock_matrix()

    def compute_energy(self):
        r"""Compute general Hartree-Fock energy, given by

        .. math:: E = D_{\beta \alpha} h_{\alpha \beta}
            + \frac{1}{2} D_{\gamma\alpha} D_{\delta\beta}
            u^{\alpha\beta}_{\gamma\delta},

        where :math:`D_{\beta\alpha}` are the matrix elements of the density
        matrix, :math:`h_{\alpha\beta}` the one-body Hamiltonian matrix
        elements of the atomic orbitals, and
        :math:`u^{\alpha\beta}_{\gamma\delta}` the antisymmetric two-body
        Hamiltonian matrix elements of the atomic orbitals.

        Returns
        -------
        complex
            The Hartree-Fock energy for general spin-orbitals.
        """

        return compute_general_hf_energy(
            self.density_matrix,
            self.system.h,
            self.system.u,
            self.system.nuclear_repulsion_energy,
            self.np,
        )

    def compute_one_body_density_matrix(self):
        r"""Compute the one-body density matrix from the occupied coefficient
        matrix. This is done by

        .. math:: \rho^{q}_{p} = \langle\Phi\rvert
                \hat{c}^{\dagger}_{p} \hat{c}_{q}
            \lvert\Phi\rangle
            = \delta_{p \in o}
            C^{*}_{\alpha p} s_{\alpha \beta} C_{\beta q},

        where :math:`\delta_{p \in o}` denotes occupied indices :math:`o = {1,
        \dots, N}`, :math:`s_{\alpha\beta}` are the overlap integrals of the
        atomic orbitals, and :math:`C_{\alpha p}` the coefficient matrix. In
        case of converged self consistent iterations, the one-body density
        matrix should yield the identity along the occupied indices and zero
        elsewhere.

        Returns
        -------
        np.ndarray
            The one-body density matrix :math:`\rho^{q}_{p}`.
        """

        np = self.np

        o = self.system.o

        rho_qp = np.zeros_like(self.system.h)
        rho_qp[o, o] = self._C[:, o].conj().T @ self.system.s @ self._C[:, o]

        return rho_qp

    def compute_particle_density(self):
        r"""Function computing the particle density :math:`\rho(x)` defined
        by

        .. math:: \rho(x) = \phi^{*}_{q}(x) \rho^{q}_{p} \phi_{p}(x),

        where :math:`\phi_p(x)` are the single-particle functions,
        :math:`\rho^{q}_{p}` the one-body density matrix, and :math:`x` some
        coordinate space. Note the use of the Einstein summation convention in
        the above expression.

        Returns
        -------
        np.ndarray
            Particle density on the same grid as the single-particle functions.
        """

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
        r"""Function computing the residual of the density matrices for the
        convergence test of the self consistent field procedure. That is,

        .. math:: \rvert\rvert \Delta \mathbf{D} \lvert\lvert_{F}
            = \rvert\rvert
                \mathbf{D}^{(i + 1)} - \mathbf{D}^{(i)}
            \lvert\lvert_{F}

        where :math:`\mathbf{D}^{(i)}` is the density matrix at iteration
        :math:`i` of the self consistent procedure.

        Note that this function works for both single density matrices, and a
        list of density matrices. This lets the ``UHF``-class reuse this
        procedure.

        Parameters
        ----------
        prev_density_matrix : np.ndarray, list
            The density matrices at the previous timestep :math:`i`.
        density_matrix : np.ndarray, list
            Density matrices at the current timestep :math:`i + 1`.

        Returns
        -------
        list
            The residuals from given density matrices.
        """
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
        r"""Compute the Hartree-Fock ground state using self consistent field
        iterations.

        Parameters
        ----------
        max_iterations : int
            The maximum number of self consistent iterations. Default is
            ``100``.
        tol : float
            The convergence tolerance for the energy and density residuals.
            Default is ``1e-4``.
        change_system_basis : bool
            Toggle whether or not to change to the Hartree-Fock basis after the
            self consistent iterations have converged.
        **mixer_kwargs : dict
            Keyword arguments to the Fock matrix mixer class.
        """

        if not "np" in mixer_kwargs:
            mixer_kwargs["np"] = self.np

        self.setup_mixer(**mixer_kwargs)
        self.compute_initial_guess()

        energy_residual = 100
        density_residual = [100]
        energy = self.compute_energy()

        for i in range(max_iterations):
            if self.verbose:
                print(
                    f"{self.__class__.__name__} energy: "
                    + f"{energy} @ iteration: {i}\t"
                    + f"residual: {density_residual}"
                )

            if (
                all(d_residual < tol for d_residual in density_residual)
                and energy_residual < tol
            ):
                break

            self.prev_density_matrix = self.density_matrix
            self.compute_scf_iteration()

            density_residual = self.compute_density_residual(
                self.prev_density_matrix, self.density_matrix
            )

            energy_prev = energy
            energy = self.compute_energy()

            energy_residual = abs(energy - energy_prev)

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
        """Function changing the system basis to the Hartree-Fock basis."""
        if self.verbose:
            print(f"Changing to {self.__class__.__name__} basis")

        self.system.change_basis(self._C)
        self._C = self.np.identity(len(self._C))
        self.density_matrix = self.build_density_matrix()
        self.fock_matrix = self.build_fock_matrix()

    @property
    def C(self):
        return self._C

    @property
    def epsilon(self):
        return self._epsilon

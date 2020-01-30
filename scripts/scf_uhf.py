import numpy as np
import scipy.linalg


def build_density_matrix(U, num_occupied):
    """Function building a density matrix.

    Build U_occ consisting of K x N (rows x columns), where K is the number of
    basis functions in total and N is the number of occupied basis functions.

    The prodcut np.dot(U_occ, U_occ.conj().T) then builds a K x K matrix, which
    is the density matrix for a given spin direction.

    Args:
        U: A two-dimensional NumPy ndarray found by solving the generalized
            eigenvalue problem for the Roothan-Hall and/or the Pople-Nesbet
            equations.
        num_occupied: The number of occupied states in the current
            spin-direction.

    Returns:
        np.ndarray: The density matrix as a two-dimensional NumPy ndarray.
    """

    U_occ = U[:, 0:num_occupied]
    return np.dot(U_occ, U_occ.conj().T)


def calculate_energy(F_up, F_down, W, D_up, D_down):
    """Function calculating the energy sans the nuclear repulsion energy.

    Note that this function is inherently designed for the UHF-case, but can be
    used by the RHF-solution using the restrictions described in the project.
    The energy is calculated by

        E = (sum over p,q,spin) F_{pq}^{spin}D_{qp}^{spin}
            - 0.5 (sum over p,q) J(D^{up} + D^{down})_{pq}
            + 0.5 (sum over p,q,spin) K(D^{spin})_{pq},

    where the matrix elements J and K are calculated using the einsum-function
    from NumPy with corresponding index-placements as calculated in the
    project.

    Args:
        F_up: The Fock-matrix for spin-up as a two-dimensional NumPy ndarray.
        F_down: The Fock-matrix for spin-down as a two-dimensional NumPy
            ndarray.
        W: A four-dimensional NumPy ndarray with the two-body electron
            repulsion integrals as elements.
        D_up: The density matrix for spin-up of the system as a two-dimensional
            NumPy ndarray.
        D_down: The density matrix for spin-down of the system as a
            two-dimensional NumPy ndarray.

    Returns:
        float: The total energy of the system sans nuclear repulsion.
    """
    J = np.einsum("prqs,sq->pr", W, D_up + D_down)
    K_up = np.einsum("psrq,sr->pq", W, D_up)
    K_down = np.einsum("psrq,sr->pq", W, D_down)

    energy = np.einsum("pq,qp->", F_up, D_up) + np.einsum(
        "pq,qp->", F_down, D_down
    )
    energy -= 0.5 * (
        np.einsum("pq,qp->", J, D_up) + np.einsum("pq,qp->", J, D_down)
    )
    energy += 0.5 * (
        np.einsum("pq,qp->", K_up, D_up) + np.einsum("pq,qp->", K_down, D_down)
    )

    return energy


def add_noise_to_density_matrix(density_matrix, weight=0.01):
    """Function adding random noise to a denisty matrix.

    To avoid getting trapped in a RHF-solution we add some random noise to the
    density matrices for the UHF-system. This function creates a matrix of
    random noise and adds it to all the elements in the density matrix.

    Args:
        density_matrix: The density matrix of the system as a two-dimensional
            NumPy ndarray.
        weight: The fraction of the noise to be added.

    Returns:
        np.ndarray: The density matrix with noise.
    """

    X = np.random.rand(*density_matrix.shape)
    X += X.T
    density_matrix += weight * X
    return density_matrix


def build_uhf_fock_matrices(H, W, D_up, D_down):
    """Function building the Fock matrix.

    The elements of the Fock matrix is given by:

        F_{pq}^{spin} = H_{pq} + J(D^{up} + D^{down})_{pq} - K(D^{spin})_{pq},

    where we build J and K from the electron repulsion integrals (W) (using
    Mulliken notation) and the density matrix D.

        J(D)_{pq} = (pq|rs)*D_{sr} = W[p][q][r][s]*D[s][r],
        K(D)_{pq} = (ps|rq)*D_{sr} = W[p][s][r][q]*D[s][r].

    The NumPy-function einsum does this in a form which resemble the
    mathematical-product.

    Args:
        H: A two-dimensional NumPy ndarray with the one-body integrals as
            elements.
        W: A four-dimensional NumPy ndarray with the two-body electron
            repulsion integrals as elements.
        D_up: The density matrix for spin-up of the system as a two-dimensional
            NumPy ndarray.
        D_down: The density matrix for spin-down of the system as a
            two-dimensional NumPy ndarray.

    Returns:
        (np.ndarray, np.ndarray): The UHF-Fock matrices for spin-up and
            spin-down respectively.
    """

    J = np.einsum("prqs,sq->pr", W, D_up + D_down)
    K_up = np.einsum("psqr,sq->pr", W, D_up)
    K_down = np.einsum("psqr,sq->pr", W, D_down)

    F_up = H + J - K_up
    F_down = H + J - K_down

    return F_up, F_down


def scf_uhf(
    H,
    W,
    S,
    num_orbitals,
    num_occupied_up,
    num_occupied_down,
    theta=0.0001,
    tol=1e-7,
    iteration_warning=True,
):
    """The SCF-scheme for solving the UHF Pople-Nesbet equations.

    Args:
        H: A two-dimensional NumPy ndarray with the one-body integrals as
            elements.
        W: A four-dimensional NumPy ndarray with the two-body electron
            repulsion integrals as elements.
        S: A two-dimensional NumPy ndarray with the overlap between the atomic
            orbital basis as elements.
        num_orbitals: The number of atomic orbitals in the basis.
        num_occupied_up: The number of occupied atomic orbitals with spin up.
        num_occupied_down: The number of occupied atomic orbitals with spin
            down.
        theta: A parameter to toggle mixing from the previous density matrix
            into the current one.
        tol: Value defining the convergence criteria for the SCF-iterations.
        iteration_warning: A boolean-value used when the user wants the program
            to get interrupted in case of divergence.

    Returns:
        float: The UHF-energy sans the nuclear repulsion energy.
    """

    # Initialize a counter to check if the iteration diverges or gets stuck
    counter = 1

    # Solve the initial generalized eigenvalue problem using the one-body
    # integrals as an approximation to the Fock matrices and the overlap matrix
    energy_up_prev, U_up = scipy.linalg.eigh(H, S)
    energy_down_prev, U_down = scipy.linalg.eigh(H, S)

    # Get the density matrices from the occupied states in the U-matrices. In
    # the UHF-case we also add some random-noise to the denisty matrices to
    # avoid getting stuck in an RHF-solution
    D_up = add_noise_to_density_matrix(
        build_density_matrix(U_up, num_occupied_up)
    )
    D_down = add_noise_to_density_matrix(
        build_density_matrix(U_down, num_occupied_down)
    )

    # Build the Fock matrices from this system
    F_up, F_down = build_uhf_fock_matrices(H, W, D_up, D_down)

    # Solve the generalized eigenvalue problems
    energy_up, U_up = scipy.linalg.eigh(F_up, S)
    energy_down, U_down = scipy.linalg.eigh(F_down, S)

    # Loop until the solution converges, i.e., until the change in the
    # eigen-energies are lower than the threshold tol
    while (
        np.max(abs(energy_up_prev - energy_up)) > tol
        and np.max(abs(energy_down_prev - energy_down)) > tol
    ):
        # Set the previous energy
        energy_up_prev = energy_up
        energy_down_prev = energy_down

        # Calculate the new density matrices adding in a theta-fraction of the
        # previous density matrices (if the need arises)
        D_up = (1 - theta) * build_density_matrix(
            U_up, num_occupied_up
        ) + theta * D_up
        D_down = (1 - theta) * build_density_matrix(
            U_down, num_occupied_down
        ) + theta * D_down

        # Build the updated Fock matrices
        F_up, F_down = build_uhf_fock_matrices(H, W, D_up, D_down)

        # Solve the new generalized eigenvalue problems
        energy_up, U_up = scipy.linalg.eigh(F_up, S)
        energy_down, U_down = scipy.linalg.eigh(F_down, S)

        # Increment the counter
        counter += 1

        # Check if we should issue a convergence warning
        if iteration_warning and counter >= 10000:
            print("WARNING: SCF (UHF) has performed %d iterations" % counter)
            input("Hit <enter> to continue...")
            counter = 0

    # Return the SCF-energy for the UHF-solution. Note that this does _not_
    # include the nuclear repulsion energy
    return calculate_energy(F_up, F_down, W, D_up, D_down)

def build_density_matrix(C, o, np):
    C_occupied = C[:, o]

    return np.dot(C_occupied, C_occupied.conj().T)


def build_general_fock_matrix(h, u, density_matrix, np):
    return h + np.tensordot(density_matrix, u, axes=((0, 1), (3, 1)))


def build_rhf_fock_matrix(h, u, density_matrix, np):
    return (
        h
        + np.tensordot(density_matrix, u, axes=((0, 1), (3, 1)))
        - 0.5 * np.tensordot(density_matrix, u, axes=((0, 1), (2, 1)))
    )


def build_uhf_fock_matrices(h, u, density_matrix, np):
    D_a, D_b = density_matrix
    D_sum = D_a + D_b

    fock_matrix_a = np.zeros_like(h)

    fock_matrix_a += h
    fock_matrix_a += np.tensordot(D_sum, u, axes=((0, 1), (3, 1)))

    fock_matrix_b = fock_matrix_a.copy()

    fock_matrix_a -= np.tensordot(D_a, u, axes=((0, 1), (2, 1)))
    fock_matrix_b -= np.tensordot(D_b, u, axes=((0, 1), (2, 1)))

    return fock_matrix_a, fock_matrix_b


def compute_error_vector(fock_matrix, density_matrix, overlap):
    """Function computing the error vector for the DIIS extrapolation method.
    See http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project8
    for the equation.
    """
    error_vector = fock_matrix @ density_matrix @ overlap
    error_vector -= overlap @ density_matrix @ fock_matrix

    return error_vector


def create_orthogonalizer(overlap_matrix, np):
    # Equation 3.166 Szabo-Ostlund
    s, U = np.linalg.eigh(overlap_matrix)
    # Equation 3.167 Szabo-Ostlund
    X = np.dot(U / np.sqrt(s), np.conj(U).T)

    return X


def compute_general_hf_energy(
    density_matrix, h, u, nuclear_repulsion_energy, np
):
    # energy <- D_{ba} h_{ab}
    energy = np.trace(np.dot(density_matrix, h))

    # term_{bd} = 0.5 * D_{ca} u^{ab}_{cd}
    term = 0.5 * np.tensordot(density_matrix, u, axes=((0, 1), (2, 0)))
    # energy <- D_{db} term_{bd}
    energy += np.trace(np.dot(density_matrix, term))

    return energy + nuclear_repulsion_energy

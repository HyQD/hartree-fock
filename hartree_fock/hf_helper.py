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
    D_up, D_down = density_matrix
    D_sum = D_up + D_down

    fock_matrix_up = np.zeros_like(h)

    fock_matrix_up += h
    fock_matrix_up += np.tensordot(D_sum, u, axes=((0, 1), (3, 1)))

    fock_matrix_down = fock_matrix_up.copy()

    fock_matrix_up -= np.tensordot(D_up, u, axes=((0, 1), (2, 1)))
    fock_matrix_down -= np.tensordot(D_down, u, axes=((0, 1), (2, 1)))

    return fock_matrix_up, fock_matrix_down


def compute_error_vector(fock_matrix, density_matrix, overlap):
    """Function computing the error vector for the DIIS extrapolation method.
    See http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project8
    for the equation.
    """
    error_vector = fock_matrix @ density_matrix @ overlap
    error_vector -= overlap @ density_matrix @ fock_matrix

    return error_vector

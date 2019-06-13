def build_density_matrix(C, o, np):
    C_occupied = C[:, o]

    return np.dot(C_occupied, C_occupied.conj().T)


def build_general_fock_matrix(h, u, density_matrix, np):
    return h + np.tensordot(density_matrix, u, axes=((0, 1), (3, 1)))


def compute_error_vector(fock_matrix, density_matrix, overlap):
    """Function computing the error vector for the DIIS extrapolation method.
    See http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project8
    for the equation.
    """
    error_vector = fock_matrix @ density_matrix @ overlap
    error_vector -= overlap @ density_matrix @ fock_matrix

    return error_vector

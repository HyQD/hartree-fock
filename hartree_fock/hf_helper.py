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


def compute_particle_density(rho_qp, spf, np):
    rho = np.zeros(spf.shape[1:], dtype=spf.dtype)
    spf_slice = slice(0, spf.shape[0])

    for _i in np.ndindex(rho.shape):
        i = (spf_slice, *_i)
        rho[_i] += np.dot(spf[i].conj(), np.dot(rho_qp, spf[i]))

    return rho


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

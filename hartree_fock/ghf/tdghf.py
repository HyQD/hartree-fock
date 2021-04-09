from hartree_fock.tdhf import TimeDependentHartreeFock


class TDGHF(TimeDependentHartreeFock):
    def compute_energy(self, current_time, C):
        np = self.np

        self.update_hamiltonian(current_time)

        C = C.reshape(self.system.l, self.system.l)
        density_matrix = self.build_density_matrix(C)

        # energy <- D_{ba} h_{ab}
        energy = np.trace(np.dot(density_matrix, self.h))

        # term_{bd} = 0.5 * D_{ca} u^{ab}_{cd}
        term = 0.5 * np.tensordot(density_matrix, self.u, axes=((0, 1), (2, 0)))
        # energy <- D_{db} term_{bd}
        energy += np.trace(np.dot(density_matrix, term))

        return energy + self.system.nuclear_repulsion_energy

    def compute_one_body_expectation_value(self, current_time, C, mat):
        D = self.compute_one_body_density_matrix(current_time, C)
        return self.np.trace(self.np.dot(D, mat))

    def compute_one_body_density_matrix(self, current_time, C):
        C = C.reshape(self.system.l, self.system.l)

        return self.build_density_matrix(C)

    def compute_overlap(self, current_time, C_a, C_b):
        C_a = C_a.reshape(self.system.l, self.system.l)
        C_b = C_b.reshape(self.system.l, self.system.l)

        S_t = self.np.einsum(
            "ki,kj->ij", C_a[:, self.system.o].conj(), C_b[:, self.system.o]
        )
        return self.np.abs(self.np.linalg.det(S_t)) ** 2

    def build_density_matrix(self, C):
        D = self.np.einsum(
            "ui,vi->uv", C[:, self.system.o], C[:, self.system.o].conj()
        )
        return D

    def build_fock_matrix(self, h, u, density_matrix):
        F = self.h + self.np.einsum("ls,usvl->uv", density_matrix, self.u)
        return F

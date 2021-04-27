from hartree_fock.tdhf import TimeDependentHartreeFock


class TDRHF(TimeDependentHartreeFock):
    def compute_energy(self, current_time, C):
        np = self.np

        self.update_hamiltonian(current_time)

        C = C.reshape(self.system.l, self.system.l)
        density_matrix = self.build_density_matrix(C)
        F = self.build_fock_matrix(self.h, self.u, density_matrix)

        energy = self.np.trace(self.np.dot(density_matrix, self.h))
        energy += self.np.trace(self.np.dot(density_matrix, F))

        return 0.5 * energy + self.system.nuclear_repulsion_energy

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
        return self.np.abs(self.np.linalg.det(S_t) ** 2) ** 2

    def build_density_matrix(self, C):
        D = 2 * self.np.einsum(
            "ui,vi->uv", C[:, self.system.o], C[:, self.system.o].conj()
        )
        return D

    def build_fock_matrix(self, h, u, density_matrix):
        F = (
            self.h
            + self.np.einsum("ls,usvl->uv", density_matrix, self.u)
            - 0.5 * self.np.einsum("ls,uslv->uv", density_matrix, self.u)
        )
        return F

from hartree_fock.hf import HartreeFock


class GHF(HartreeFock):
    def build_density_matrix(self, C):
        P = self.np.einsum(
            "ui,vi->uv", C[:, self.system.o], C[:, self.system.o].conj()
        )
        return P

    def build_fock_matrix(self, P):
        F = self.system.h.copy()
        F += self.np.einsum("ls,usvl->uv", P, self.system.u)
        return F

    def _compute_energy(self, P, F):
        e_ghf = self.np.trace(self.np.dot(P, self.system.h))
        e_ghf += self.np.trace(self.np.dot(P, F))
        return 0.5 * e_ghf + self.system.nuclear_repulsion_energy

    def compute_two_body_density_matrix(self):
        rho_qp = self.compute_one_body_density_matrix()
        rho_rspq = -self.np.einsum(
            "sp, rq -> rspq", rho_qp, rho_qp
        ) + self.np.einsum("rp, sq -> rspq", rho_qp, rho_qp)

        return rho_rspq

    def compute_one_body_expectation_value(self, mat):
        return super().compute_one_body_expectation_value(mat)

    def compute_two_body_expectation_value(self, op, asym=True):
        rho_rspq = self.compute_two_body_density_matrix()

        return (
            0.5
            * (0.5 if asym else 1)
            * self.np.einsum("pqrs, rspq ->", op, rho_rspq)
        )

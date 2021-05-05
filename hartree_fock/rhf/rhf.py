from hartree_fock.hf import HartreeFock


class RHF(HartreeFock):
    def build_density_matrix(self, C):
        P = 2 * self.np.einsum(
            "ui,vi->uv", C[:, self.system.o], C[:, self.system.o].conj()
        )
        return P

    def build_fock_matrix(self, P):
        F = self.system.h.copy()
        F += self.np.einsum("ls,usvl->uv", P, self.system.u)
        F -= 0.5 * self.np.einsum("ls,uslv->uv", P, self.system.u)
        return F

    def _compute_energy(self, P, F):
        e_rhf = self.np.trace(self.np.dot(P, self.system.h))
        e_rhf += self.np.trace(self.np.dot(P, F))
        return 0.5 * e_rhf + self.system.nuclear_repulsion_energy

    def compute_two_body_density_matrix(self):
        rho_qp = self.compute_one_body_density_matrix()

        rho_rspq = self.np.einsum(
            "rp, sq -> rspq", rho_qp, rho_qp
        ) - 0.5 * self.np.einsum("sp, rq -> rspq", rho_qp, rho_qp)

        return rho_rspq

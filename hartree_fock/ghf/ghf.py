from hartree_fock.hf import HartreeFock

class GHF(HartreeFock):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)
    
    def build_density_matrix(self,C):
        P = self.np.einsum("ui,vi->uv", C[:, self.system.o], C[:, self.system.o].conj())
        return P

    def build_fock_matrix(self,P):
        F = self.system.h.copy()
        F += self.np.einsum("ls,usvl->uv", P, self.system.u)
        return F

    def compute_energy(self,P,F):
        e_ghf = self.np.trace(self.np.dot(P, self.system.h))
        e_ghf += self.np.trace(self.np.dot(P, F))
        return 0.5*e_ghf

from hartree_fock.hf import HartreeFock


class UHF(HartreeFock):
    def __init__(self, system, **kwargs):
        super().__init__(system, **kwargs)

    def build_density_matrix(self, C):
        pass

    def build_fock_matrix(self, P):
        pass

    def compute_energy(self, P, F):
        pass

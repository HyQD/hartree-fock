from hartree_fock import HartreeFock
from hartree_fock.hf_helper import build_density_matrix, build_rhf_fock_matrix


class RHF(HartreeFock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RHF assumes double occupancies of all orbitals
        self.o = slice(0, self.system.n // 2)

    def compute_one_body_density_matrix(self):
        return 2 * super().compute_one_body_density_matrix()

    def build_density_matrix(self):
        return 2 * build_density_matrix(self._C, self.o, self.np)

    def build_fock_matrix(self):
        return build_rhf_fock_matrix(
            self.system.h, self.system.u, self.density_matrix, self.np
        )

    def compute_energy(self):
        np = self.np

        # term_{ab} <- D_{dc} I^{ac}_{bd}
        term = np.tensordot(
            self.density_matrix, self.system.u, axes=((0, 1), (3, 1))
        )
        # term_{ab} <- term_{ab} -0.5 * D_{dc} I^{ac}_{db}
        term -= 0.5 * np.tensordot(
            self.density_matrix, self.system.u, axes=((0, 1), (2, 1))
        )

        # term_{ab} <- h_{ab} + 0.5 term_{ab}
        term = self.system.h + 0.5 * term

        # energy = D_{ba} term_{ab}
        energy = np.trace(np.dot(term, self.density_matrix))

        return energy + self.system.nuclear_repulsion_energy

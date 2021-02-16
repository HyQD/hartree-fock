from hartree_fock.tdhf import TimeDependentHartreeFock


class TDGHF(TimeDependentHartreeFock):
    def compute_energy(self, current_time, C):
        np = self.np

        C = C.reshape(self.system.l, self.system.l)
        density_matrix = self.build_density_matrix(C, self.o)

        # energy <- D_{ba} h_{ab}
        energy = np.trace(np.dot(density_matrix, self.h))

        # term_{bd} = 0.5 * D_{ca} u^{ab}_{cd}
        term = 0.5 * np.tensordot(density_matrix, self.u, axes=((0, 1), (2, 0)))
        # energy <- D_{db} term_{bd}
        energy += np.trace(np.dot(density_matrix, term))

        return energy + self.system.nuclear_repulsion_energy

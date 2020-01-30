import numpy as np

from quantum_systems import ODQD, construct_pyscf_system_ao
from hartree_fock import GHF, RHF, UHF

from scf_uhf import scf_uhf


n = 2
l = 12

# system = ODQD(n, l, 11, 201)
# system.setup_system(add_spin=False, anti_symmetrize=False)
system = construct_pyscf_system_ao("he", add_spin=False, anti_symmetrize=False)

rhf_system = system.copy_system()
uhf_system = system.copy_system()
u_2 = system.copy_system()
system.change_to_spin_orbital_basis(anti_symmetrize=True)
system_2 = system.copy_system()

ghf = GHF(system, verbose=True)
rhf = RHF(rhf_system, verbose=True)
uhf = UHF(uhf_system, verbose=True)

ghf.compute_ground_state(max_iterations=0)
rhf.compute_ground_state(max_iterations=0)
uhf.compute_ground_state(max_iterations=0)

system_2.change_to_hf_basis(verbose=True)

print(
    (
        f"SCF UHF: {scf_uhf(u_2.h, u_2.u.transpose(0, 2, 1, 3), u_2.s, u_2.n, u_2.n_a, u_2.n_b) + u_2.nuclear_repulsion_energy}"
    )
)

print(f"GHF energy: {ghf.compute_energy()}")
print(f"RHF energy: {rhf.compute_energy()}")
print(f"UHF energy: {uhf.compute_energy()}")

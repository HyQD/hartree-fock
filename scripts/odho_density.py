import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import ODQD
from hartree_fock import GHF, RHF, UHF
from hartree_fock.mix import DIIS


n = 3
l = 20

omega = 0.25

odqd = ODQD(n, l, 10, 201)
odqd.setup_system(potential=ODQD.HOPotential(omega=omega))

ghf = GHF(odqd, verbose=True)  # , mixer=DIIS)
ghf.compute_ground_state()

# odqd_rhf = ODQD(n, l, 10, 201)
# odqd_rhf.setup_system(potential=ODQD.HOPotential(omega=omega), add_spin=False)
#
# rhf = RHF(odqd_rhf, verbose=True)  # , mixer=DIIS)
# rhf.compute_ground_state()

odqd_uhf = ODQD(n, l, 10, 201)
odqd_uhf.setup_system(potential=ODQD.HOPotential(omega=omega), add_spin=False)

uhf = UHF(odqd_uhf, verbose=True)
uhf.compute_ground_state()

plt.plot(odqd.grid, ghf.compute_particle_density().real, label="GHF")
# plt.plot(odqd_rhf.grid, rhf.compute_particle_density().real, label="RHF")
plt.plot(
    odqd_uhf.grid,
    uhf.compute_particle_density(direction="up").real
    + uhf.compute_particle_density(direction="down").real,
    label="UHF",
)
plt.plot(
    odqd_uhf.grid,
    uhf.compute_particle_density(direction="up").real,
    label="UHF (up)",
)
plt.plot(
    odqd_uhf.grid,
    uhf.compute_particle_density(direction="down").real,
    label="UHF (down)",
)
plt.legend()
plt.show()

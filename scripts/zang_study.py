import os

import numpy as np
import matplotlib.pyplot as plt


from quantum_systems import ODQD, GeneralOrbitalSystem
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from hartree_fock import GHF, TDGHF


n = 2
l = 10
grid_length = 10
num_grid_points = 2001

alpha = 1
a = 0.25

omega = 0.25

potential = ODQD.HOPotential(omega)

odqd = ODQD(
    l=l,
    grid_length=grid_length,
    num_grid_points=num_grid_points,
    alpha=alpha,
    a=a,
    potential=potential,
)
system = GeneralOrbitalSystem(n, odqd)

ghf = GHF(system, verbose=True).compute_ground_state(tol=1e-7)

test_rho = np.loadtxt(os.path.join("tests", "dat", "rho_tdhf_real.dat"))

fig, ax = plt.subplots(1, figsize=(16, 10))
img = plt.imread(os.path.join("scripts", "zang-density.png"))
ax.imshow(img, extent=(-6, 6, 0, 0.4), aspect="auto")
ax.plot(
    system._basis_set.grid, ghf.compute_particle_density().real, label="GHF"
)
ax.plot(test_rho[:, 0], test_rho[:, 1], "--", label="Test")
ax.axis([-6, 6, 0, 0.4])
ax.grid()
ax.legend(loc="best")
plt.show()

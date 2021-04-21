import os

import numpy as np
import matplotlib.pyplot as plt


from quantum_systems import ODQD, GeneralOrbitalSystem, SpatialOrbitalSystem
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from hartree_fock import GHF, TDGHF, RHF


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

gos = GeneralOrbitalSystem(n, odqd.copy_basis())
gos_2 = GeneralOrbitalSystem(n, odqd.copy_basis())
spas = SpatialOrbitalSystem(n, odqd)

gos_2._basis_set.h = gos_2._basis_set.h + 0.1 * gos_2._basis_set.spin_z


# for i in range(0, gos.l, 2):
#     norm = gos.s[i, i] + gos.s[i + 1, i + 1]
#     gos._basis_set.spf[i] = gos.spf[i] / norm
#     gos._basis_set.spf[i + 1] = gos.spf[i + 1] / norm


colors = ["b", "r", "g", "y", "c", "m", "k"]

plt.figure()
plt.title("HO (GOS) basis")
plt.plot(gos._basis_set.grid, potential(gos._basis_set.grid))

for i in range(0, gos.l, 2):
    color = colors[(i // 2) % len(colors)]
    norm = 1 / np.abs(gos.s[i, i] + gos.s[i + 1, i + 1])

    plt.plot(
        gos._basis_set.grid,
        norm ** 2 * (np.abs(gos.spf[i]) + np.abs(gos.spf[i + 1])) ** 2
        # np.abs(norm * (gos.spf[i] + gos.spf[i + 1])) ** 2
        + norm * (gos.h[i, i] + gos.h[i + 1, i + 1]).real,
        "-" + color,
        label=r"$\psi_{" + f"{i // 2}" + r"}$ (GOS)",
    )

plt.grid()
plt.legend()


plt.figure()
plt.title("HO (SpaS) basis")
plt.plot(spas._basis_set.grid, potential(spas._basis_set.grid))

for i in range(spas.l):
    color = colors[i % len(colors)]
    plt.plot(
        spas._basis_set.grid,
        np.abs(spas.spf[i]) ** 2 + spas.h[i, i].real,
        "-" + color,
        label=r"$\psi_{" + f"{i}" + r"}$",
    )

plt.grid()
plt.legend()

plt.figure()
plt.title("HO basis compared")
plt.plot(gos._basis_set.grid, potential(gos._basis_set.grid))

for i in range(0, gos.l, 2):
    norm = 1 / np.abs(gos.s[i, i] + gos.s[i + 1, i + 1])

    plt.plot(
        gos._basis_set.grid,
        norm ** 2 * (np.abs(gos.spf[i]) + np.abs(gos.spf[i + 1])) ** 2
        # np.abs(norm * (gos.spf[i] + gos.spf[i + 1])) ** 2
        + norm * (gos.h[i, i] + gos.h[i + 1, i + 1]).real,
        label=r"$\psi_{" + f"{i // 2}" + r"}$ (GOS)",
    )

    plt.plot(
        spas._basis_set.grid,
        np.abs(spas.spf[i // 2]) ** 2 + spas.h[i // 2, i // 2].real,
        "--",
        label=r"$\psi_{" + f"{i // 2}" + r"}$ (SpaS)",
    )


plt.grid()
plt.legend()

ghf = GHF(gos, verbose=True).compute_ground_state(tol=1e-7)
ghf_2 = GHF(gos_2, verbose=True).compute_ground_state(tol=1e-7)
rhf = RHF(spas, verbose=True).compute_ground_state(tol=1e-7)

plt.figure()
plt.title("GHF 2 basis")
plt.plot(gos_2._basis_set.grid, potential(gos._basis_set.grid))

for i in range(0, gos_2.l):
    color = colors[(i // 2) % len(colors)]
    # norm = 1 / np.abs(gos.s[i, i] + gos.s[i + 1, i + 1])

    plt.plot(
        gos_2._basis_set.grid,
        np.abs(gos_2.spf[i]) ** 2 + gos_2.h[i, i].real,
        # norm ** 2 * (np.abs(gos.spf[i]) + np.abs(gos.spf[i + 1])) ** 2
        # np.abs(norm * (gos.spf[i] + gos.spf[i + 1])) ** 2
        # + norm * (gos.h[i, i] + gos.h[i + 1, i + 1]).real,
        "-" + color,
        label=r"$\psi_{" + f"{i}" + r"}$",
    )

plt.grid()
plt.legend()


plt.figure()
plt.title("GHF basis (spin-traced)")
plt.plot(gos._basis_set.grid, potential(gos._basis_set.grid))

for i in range(0, gos.l, 2):
    color = colors[(i // 2) % len(colors)]
    norm = 1 / np.abs(gos.s[i, i] + gos.s[i + 1, i + 1])

    plt.plot(
        gos._basis_set.grid,
        norm ** 2 * (np.abs(gos.spf[i]) + np.abs(gos.spf[i + 1])) ** 2
        # np.abs(norm * (gos.spf[i] + gos.spf[i + 1])) ** 2
        + norm * (gos.h[i, i] + gos.h[i + 1, i + 1]).real,
        "-" + color,
        label=r"$\psi_{" + f"{i // 2}" + r"}$",
    )

plt.grid()
plt.legend()

plt.figure()
plt.title("GHF basis (full)")
plt.plot(gos._basis_set.grid, potential(gos._basis_set.grid))

for i in range(0, gos.l):
    color = colors[(i // 2) % len(colors)]
    # norm = 1 / np.abs(gos.s[i, i] + gos.s[i + 1, i + 1])

    plt.plot(
        gos._basis_set.grid,
        np.abs(gos.spf[i]) ** 2
        # (np.abs(gos.spf[i]) + np.abs(gos.spf[i + 1])) ** 2
        # np.abs(norm * (gos.spf[i] + gos.spf[i + 1])) ** 2
        + gos.h[i, i].real,
        ("-" if i % 2 == 0 else "--") + color,
        label=r"$\psi_{" + f"{i // 2}" + r"}$",
    )

plt.grid()
plt.legend()

# plt.figure()
# plt.title("GHF basis (Re and Im) (spin-traced)")
# plt.plot(gos._basis_set.grid, potential(gos._basis_set.grid))
#
#
# for i in range(0, gos.l, 2):
#     color = colors[(i // 2) % len(colors)]
#
#     norm = 1 / np.abs(gos.s[i, i] + gos.s[i + 1, i + 1])
#
#     plt.plot(
#         gos._basis_set.grid,
#         norm * (gos.spf[i] + gos.spf[i + 1]).real
#         # norm ** 2 * (np.abs(gos.spf[i]) + np.abs(gos.spf[i + 1])) ** 2
#         # np.abs(norm * (gos.spf[i] + gos.spf[i + 1])) ** 2
#         + norm * (gos.h[i, i] + gos.h[i + 1, i + 1]).real,
#         "-" + color,
#         label=r"$\Re{\psi_{" + f"{i // 2}" + r"}}$",
#     )
#
#     plt.plot(
#         gos._basis_set.grid,
#         norm * (gos.spf[i] + gos.spf[i + 1]).imag
#         # norm ** 2 * (np.abs(gos.spf[i]) + np.abs(gos.spf[i + 1])) ** 2
#         # np.abs(norm * (gos.spf[i] + gos.spf[i + 1])) ** 2
#         + norm * (gos.h[i, i] + gos.h[i + 1, i + 1]).real,
#         "--" + color,
#         label=r"$\Im{\psi_{" + f"{i // 2}" + r"}}$",
#     )
#
# plt.grid()
# plt.legend()


plt.figure()
plt.title("RHF basis")
plt.plot(spas._basis_set.grid, potential(spas._basis_set.grid))

for i in range(0, spas.l):
    color = colors[i % len(colors)]
    plt.plot(
        spas._basis_set.grid,
        np.abs(spas.spf[i]) ** 2 + spas.h[i, i].real,
        "-" + color,
        label=r"$\psi_{" + f"{i}" + r"}$",
    )

plt.grid()
plt.legend()


plt.figure()
plt.title("GHF and RHF basis compared")
plt.plot(gos._basis_set.grid, potential(gos._basis_set.grid))

for i in range(0, gos.l, 2):
    norm = 1 / np.abs(gos.s[i, i] + gos.s[i + 1, i + 1])

    plt.plot(
        gos._basis_set.grid,
        norm ** 2 * (np.abs(gos.spf[i]) + np.abs(gos.spf[i + 1])) ** 2
        # np.abs(norm * (gos.spf[i] + gos.spf[i + 1])) ** 2
        + norm * (gos.h[i, i] + gos.h[i + 1, i + 1]).real,
        label=r"$\psi_{" + f"{i // 2}" + r"}$ (GHF)",
    )

    plt.plot(
        spas._basis_set.grid,
        np.abs(spas.spf[i // 2]) ** 2 + spas.h[i // 2, i // 2].real,
        "--",
        label=r"$\psi_{" + f"{i // 2}" + r"}$ (RHF)",
    )


plt.grid()
plt.legend()


rho_rhf_qp = rhf.compute_one_body_density_matrix()
rho_ghf_qp = ghf.compute_one_body_density_matrix()

print(rho_rhf_qp)
print("=" * 100)
print(rho_ghf_qp)


rho_rhf = rhf.compute_particle_density().real

rho_ghf_gp_red = np.abs(gos.spf[0]) ** 2 + np.abs(gos.spf[1]) ** 2

print(np.trapz(rho_rhf, spas._basis_set.grid))
print(np.trapz(rho_ghf_gp_red, gos._basis_set.grid))


test_rho = np.loadtxt(os.path.join("tests", "dat", "rho_tdhf_real.dat"))

fig, ax = plt.subplots(1, figsize=(16, 10))
img = plt.imread(os.path.join("scripts", "zang-density.png"))
ax.imshow(img, extent=(-6, 6, 0, 0.4), aspect="auto")
ax.plot(spas._basis_set.grid, rho_rhf, label="RHF")
ax.plot(gos._basis_set.grid, rho_ghf_gp_red, label="GHF")
ax.plot(test_rho[:, 0], test_rho[:, 1], "--", label="Test")
ax.axis([-6, 6, 0, 0.4])
ax.grid()
ax.legend(loc="best")
plt.show()

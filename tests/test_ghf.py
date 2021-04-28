from hartree_fock.ghf import GHF
from quantum_systems import (
    TwoDimensionalHarmonicOscillator,
    GeneralOrbitalSystem,
)
from quantum_systems import construct_pyscf_system_ao
import numpy as np


def test_lih_ghf():

    molecule = "li 0.0 0.0 0.0; h 3.08 0.0 0.0"

    basis = "cc-pvdz"

    system = construct_pyscf_system_ao(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=True,
        anti_symmetrize=True,
    )

    ghf = GHF(system, verbose=False)
    ghf.compute_ground_state(tol=1e-12, change_system_basis=False)

    e_ghf = ghf.total_energy.real

    dip_mom_ghf_x = ghf.compute_one_body_expectation_value(
        system.dipole_moment[0]
    ).real
    dip_mom_ghf_y = ghf.compute_one_body_expectation_value(
        system.dipole_moment[1]
    ).real
    dip_mom_ghf_z = ghf.compute_one_body_expectation_value(
        system.dipole_moment[2]
    ).real

    import pyscf

    # Build molecule in AO-basis
    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.unit = "bohr"
    mol.build(atom=molecule, basis=basis)
    nuclear_repulsion_energy = mol.energy_nuc()

    hf = pyscf.scf.RHF(mol)
    hf.conv_tol = 1e-12
    hf.kernel()
    Cocc = hf.mo_coeff[:, : system.n // 2]
    D = 2 * np.einsum("sj,rj->sr", Cocc.conj(), Cocc)
    dipole_integrals = -mol.intor("int1e_r").reshape(
        3, system.l // 2, system.l // 2
    )

    dip_mom_pyscf_x = np.trace(np.dot(D, dipole_integrals[0]))
    dip_mom_pyscf_y = np.trace(np.dot(D, dipole_integrals[1]))
    dip_mom_pyscf_z = np.trace(np.dot(D, dipole_integrals[2]))

    assert abs(dip_mom_pyscf_x - dip_mom_ghf_x) < 1e-5
    assert abs(dip_mom_pyscf_y - dip_mom_ghf_y) < 1e-5
    assert abs(dip_mom_pyscf_z - dip_mom_ghf_z) < 1e-5
    assert abs(e_ghf - hf.e_tot) < 1e-5


def test_h2o_ghf():

    r = 1.871
    molecule = f"O; H, 1, {r};  H 1 {r} 2 100.0"
    basis = "sto-3g"

    system = construct_pyscf_system_ao(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=True,
        anti_symmetrize=True,
    )

    ghf = GHF(system, verbose=True)
    ghf.compute_ground_state(tol=1e-12)
    e_hf_pyscf = -74.965_900_173_175_2

    assert abs(ghf.total_energy - e_hf_pyscf) < 1e-10


def test_h2_hf():
    molecule = f"h 0.0 0.0 -0.69485; h 0.0 0.0 0.69485"
    basis = "6-311++Gss"

    system = construct_pyscf_system_ao(molecule, basis=basis)

    hf = GHF(system, verbose=True)
    hf.compute_ground_state(tol=1e-10, change_system_basis=True)

    assert abs(hf.compute_energy() - (-1.1322)) < 1e-3

    n = system.n
    l = system.l

    for i in range(l):
        C_i = hf.C[:, i]
        for j in range(i + 1, l):
            C_j = hf.C[:, j]

            assert abs(C_i.conj().T @ system.s @ C_j) < 1e-8, f"({i}, {j})"

    rho_qp = hf.compute_one_body_density_matrix()

    assert abs(np.trace(rho_qp) - n) < 1e-8

    np.testing.assert_allclose(
        rho_qp[system.o, system.o], np.identity(n), atol=1e-10
    )
    np.testing.assert_allclose(
        rho_qp[system.v, system.v],
        np.zeros((l - n, l - n), dtype=np.complex128),
        atol=1e-10,
    )


def test_tdho_hf():
    n = 2
    l = 12
    radius = 10
    num_grid_points = 401

    tdho = GeneralOrbitalSystem(
        2, TwoDimensionalHarmonicOscillator(l // 2, radius, num_grid_points)
    )

    hf = GHF(tdho, verbose=True)

    hf.compute_ground_state(tol=1e-10)  # , num_vecs=20, max_iterations=1000)

    assert abs(hf.compute_energy() - 3.162_691) < 1e-6

    for i in range(l):
        C_i = hf.C[:, i]
        for j in range(i + 1, l):
            C_j = hf.C[:, j]

            assert abs(C_i.conj() @ C_j) < 1e-8

    rho_qp = hf.compute_one_body_density_matrix()
    assert abs(np.trace(rho_qp) - n) < 1e-8

    np.testing.assert_allclose(
        rho_qp, np.diag(np.append(np.ones(n), np.zeros(l - n))), atol=1e-14
    )

    rho = hf.compute_particle_density()

from hartree_fock.rhf import RHF
from quantum_systems import (
    TwoDimensionalHarmonicOscillator,
    SpatialOrbitalSystem,
)
from quantum_systems import construct_pyscf_system_ao
import numpy as np
from hartree_fock.mix import AlphaMixer, DIIS
import pytest


def test_rhf():

    molecule = "li 0.0 0.0 0.0; h 3.08 0.0 0.0"

    basis = "cc-pvdz"

    system = construct_pyscf_system_ao(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    rhf = RHF(system, verbose=False)
    rhf.compute_ground_state(tol=1e-12, change_system_basis=False)

    e_rhf = rhf.total_energy.real
    e_rhf_2 = (
        rhf.compute_one_body_expectation_value(system.h)
        + rhf.compute_two_body_expectation_value(system.u)
        + system.nuclear_repulsion_energy
    )

    assert abs(e_rhf - e_rhf_2) < 1e-12

    dip_mom_rhf_x = rhf.compute_one_body_expectation_value(
        system.dipole_moment[0]
    ).real
    dip_mom_rhf_y = rhf.compute_one_body_expectation_value(
        system.dipole_moment[1]
    ).real
    dip_mom_rhf_z = rhf.compute_one_body_expectation_value(
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
    Cocc = hf.mo_coeff[:, : system.n]
    D = 2 * np.einsum("sj,rj->sr", Cocc.conj(), Cocc)
    dipole_integrals = -mol.intor("int1e_r").reshape(3, system.l, system.l)

    dip_mom_pyscf_x = np.trace(np.dot(D, dipole_integrals[0]))
    dip_mom_pyscf_y = np.trace(np.dot(D, dipole_integrals[1]))
    dip_mom_pyscf_z = np.trace(np.dot(D, dipole_integrals[2]))

    assert abs(dip_mom_pyscf_x - dip_mom_rhf_x) < 1e-5
    assert abs(dip_mom_pyscf_y - dip_mom_rhf_y) < 1e-5
    assert abs(dip_mom_pyscf_z - dip_mom_rhf_z) < 1e-5
    assert abs(e_rhf - hf.e_tot) < 1e-5


def test_h2o_rhf():

    r = 1.871
    molecule = f"O; H, 1, {r};  H 1 {r} 2 100.0"
    basis = "sto-3g"

    system = construct_pyscf_system_ao(
        molecule,
        basis=basis,
        np=np,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    rhf = RHF(system, verbose=True).compute_ground_state(tol=1e-12)

    e_rhf = rhf.compute_energy().real
    e_rhf_2 = (
        rhf.compute_one_body_expectation_value(system.h)
        + rhf.compute_two_body_expectation_value(system.u)
        + system.nuclear_repulsion_energy
    ).real
    e_hf_pyscf = -74.965_900_173_175_2

    assert abs(e_rhf - e_rhf_2) < 1e-12
    assert abs(e_rhf - e_hf_pyscf) < 1e-10


def test_tdho_rhf():
    n = 2
    l = 12
    radius = 10
    num_grid_points = 401

    tdho = SpatialOrbitalSystem(
        2, TwoDimensionalHarmonicOscillator(l, radius, num_grid_points)
    )

    rhf = RHF(tdho, verbose=True).compute_ground_state(tol=1e-10)

    assert abs(rhf.compute_energy() - 3.162_691) < 1e-6

import numpy as np

from hartree_fock import GHF
from quantum_systems import TwoDimensionalHarmonicOscillator


def test_tdho_hf():
    n = 2
    l = 12
    radius = 10
    num_grid_points = 401

    tdho = TwoDimensionalHarmonicOscillator(n, l, radius, num_grid_points)
    tdho.setup_system()

    hf = GHF(tdho, verbose=True)

    hf.compute_ground_state(tol=1e-10)  # , num_vecs=20, max_iterations=1000)

    assert abs(hf.compute_energy() - 3.162691) < 1e-6

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


def test_h2_hf():
    from quantum_systems import construct_pyscf_system_ao

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

import psi4
from hartree_fock import UHF
from hartree_fock.mix import DIIS, AlphaMixer
from quantum_systems import (
    TwoDimensionalHarmonicOscillator,
    construct_psi4_system,
)


def get_psi4_uhf_energy(molecule, options):
    psi4.core.be_quiet()

    if "reference" not in options:
        options["reference"] = "uhf"

    psi4.set_options(options)
    mol = psi4.geometry(molecule)

    wavefunction = psi4.core.Wavefunction.build(
        mol, psi4.core.get_global_option("BASIS")
    )

    # Perform a SCF calculation based on the options.
    uhf_energy = psi4.energy("SCF")

    return uhf_energy


def test_tdho_uhf():
    n = 2
    l = 12
    radius = 10
    num_grid_points = 401

    tdho = TwoDimensionalHarmonicOscillator(n, l, radius, num_grid_points)
    tdho.setup_system(add_spin=False, anti_symmetrize=False)

    uhf = UHF(tdho, verbose=True)

    uhf.compute_ground_state(tol=1e-10)

    assert abs(uhf.compute_energy() - 3.162691) < 1e-6


def test_triplet_O2_uhf():
    # Triplet O2
    mol = """
        0 3
        O
        O 1 1.2
        symmetry c1
        """

    options = {
        "guess": "core",
        "basis": "aug-cc-pvdz",
        "scf_type": "df",
        "e_convergence": 1e-8,
        "reference": "uhf",
    }

    psi4_energy = get_psi4_uhf_energy(mol, options)

    system = construct_psi4_system(
        mol, options, add_spin=False, anti_symmetrize=False
    )
    uhf = UHF(system, verbose=True)
    uhf.compute_ground_state(tol=1e-8)

    assert abs(uhf.compute_energy() - psi4_energy) < 1e-3

from quantum_systems import TwoDimensionalDoubleWell
from hartree_fock import HartreeFock


tddw = TwoDimensionalDoubleWell(4, 20, 10, 101)
tddw.setup_system()

print(tddw.h)

hf = HartreeFock(tddw, verbose=True)
hf.compute_ground_state()

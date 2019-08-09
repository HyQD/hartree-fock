Ground state calculations
=========================
This library provides three Hartree-Fock solvers, they are:

* Hartree-Fock solver with general spin-orbitals, ``HartreeFock``.
* The restricted Hartree-Fock method, ``RHF``.
* The unrestricted Hartree-Fock method, ``UHF``.

All classes re-use the self-consistent field procedure defined in the
``HartreeFock``-class, but implement their own methods where needed, e.g., the
construction of the Fock matrix.

.. autoclass:: hartree_fock.hf.HartreeFock
    :members:

.. autoclass:: hartree_fock.rhf.RHF
    :members:

.. autoclass:: hartree_fock.uhf.UHF
    :members:

import numpy as np
import pytest
from hartree_fock.hf_helper import build_density_matrix


@pytest.mark.skip
def test_density_matrix():
    n = 4
    l = 20

    o = slice(0, n)

    for i in range(10):
        C = np.random.random((l, l)) + 1j * np.random.random((l, l))

        D = np.einsum("pi, qi -> qp", C[:, o].conj(), C[:, o])
        D_helper = build_density_matrix(C, o, np)

        np.testing.assert_allclose(D, D_helper)

"""Collective-spin matrices used by both Bell scenarios."""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigh


def collective_spin_matrices(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Pauli-sum ``Sx`` and ``Sz`` in the spin-``N/2`` irrep."""

    if N < 1:
        raise ValueError("N must be a positive integer")

    spin = N / 2.0
    magnetic = np.arange(-spin, spin + 1.0, 1.0)
    dimension = N + 1
    raising = np.zeros((dimension, dimension), dtype=float)

    for index, value in enumerate(magnetic[:-1]):
        raising[index + 1, index] = np.sqrt(
            spin * (spin + 1.0) - value * (value + 1.0)
        )

    sx = raising + raising.T
    sz = 2.0 * np.diag(magnetic)
    return sx, sz


def min_eigpair(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    hermitian = (matrix + matrix.T.conj()) / 2.0
    values, vectors = eigh(hermitian)
    index = int(np.argmin(values))
    return float(np.real(values[index])), vectors[:, index].copy()

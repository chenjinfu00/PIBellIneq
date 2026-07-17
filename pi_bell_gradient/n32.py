"""Permutation-invariant ``(N,3,2)`` Bell-gradient model."""

from __future__ import annotations

from itertools import permutations, product
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from . import core
from .spin import collective_spin_matrices, min_eigpair


COEFFICIENT_ORDER = (
    "alpha0",
    "alpha1",
    "alpha2",
    "alpha00",
    "alpha01",
    "alpha02",
    "alpha11",
    "alpha12",
    "alpha22",
)
TARGET_RANK = len(COEFFICIENT_ORDER)


def _diag_position(setting: int) -> int:
    positions = (3, 6, 8)
    if setting not in (0, 1, 2):
        raise ValueError("setting must be 0, 1, or 2")
    return positions[setting]


def _pair_position(first: int, second: int) -> int:
    pair = tuple(sorted((first, second)))
    positions = {(0, 1): 4, (0, 2): 5, (1, 2): 7}
    if pair not in positions:
        raise ValueError("pair must be one of (0,1), (0,2), or (1,2)")
    return positions[pair]


def transform_alpha(
    alpha: Sequence[float] | np.ndarray,
    permutation: Sequence[int],
    signs: Sequence[int],
) -> np.ndarray:
    """Apply a measurement permutation and independent outcome flips."""

    coefficients = np.asarray(alpha)
    if coefficients.shape != (TARGET_RANK,):
        raise ValueError(f"alpha must have shape ({TARGET_RANK},)")
    permutation = tuple(int(value) for value in permutation)
    signs = tuple(int(value) for value in signs)
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError("permutation must contain 0, 1, and 2")
    if len(signs) != 3 or any(sign not in (-1, 1) for sign in signs):
        raise ValueError("signs must contain three values in {-1,+1}")

    inverse = [0, 0, 0]
    for old_setting, new_setting in enumerate(permutation):
        inverse[new_setting] = old_setting

    transformed = np.empty(TARGET_RANK, dtype=coefficients.dtype)
    for new_setting in range(3):
        old_setting = inverse[new_setting]
        transformed[new_setting] = signs[old_setting] * coefficients[old_setting]
        transformed[_diag_position(new_setting)] = coefficients[
            _diag_position(old_setting)
        ]

    for new_first, new_second in ((0, 1), (0, 2), (1, 2)):
        old_first = inverse[new_first]
        old_second = inverse[new_second]
        transformed[_pair_position(new_first, new_second)] = (
            signs[old_first]
            * signs[old_second]
            * coefficients[_pair_position(old_first, old_second)]
        )

    return transformed


def canonical_alpha_N32(
    alpha: Sequence[float] | np.ndarray,
    reverse_lexicographic: bool = True,
) -> dict[str, Any]:
    """Choose one representative of the 48-element N32 symmetry orbit."""

    representatives = []
    for permutation in permutations((0, 1, 2)):
        for signs in product((1, -1), repeat=3):
            transformed = transform_alpha(alpha, permutation, signs)
            key = tuple(
                transformed[::-1] if reverse_lexicographic else transformed
            )
            representatives.append((key, transformed, permutation, signs))

    _, transformed, permutation, signs = max(
        representatives,
        key=lambda item: item[0],
    )
    return {
        "alpha": transformed.copy(),
        "permutation": permutation,
        "signs": signs,
    }


# Compatibility with the name used in the exploratory notebook.
canonical_alpha_N33 = canonical_alpha_N32


def enumerate_vertices_N32(N: int) -> list[np.ndarray]:
    """Return vertices ordered as ``[S0,S1,S2,S00,S01,S02,S11,S12,S22]``."""

    return core.enumerate_pi_vertices(N, settings=3)


def quantum_value_grad(
    alpha: Sequence[float] | np.ndarray,
    N: int,
    grid_n: int = 6,
    rand_n: int = 200,
    tol: float = 1e-10,
    seed: int | None = None,
    report: bool = False,
) -> dict[str, Any]:
    """Minimize over two planar angles and return HF coefficient gradients."""

    coefficients = np.asarray(alpha, dtype=float)
    if coefficients.size != TARGET_RANK:
        raise ValueError(f"alpha must have length {TARGET_RANK}")

    a0, a1, a2, a00, a01, a02, a11, a12, a22 = coefficients
    sx, sz = collective_spin_matrices(N)
    identity = np.eye(N + 1)
    period = 2.0 * np.pi
    rng = np.random.default_rng(seed)

    def build(theta: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        theta1, theta2 = np.asarray(theta, dtype=float)
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        s0 = sz
        s1_operator = c1 * sz + s1 * sx
        s2_operator = c2 * sz + s2 * sx

        o00 = 0.5 * (s0 @ s0 - N * identity)
        o11 = 0.5 * (s1_operator @ s1_operator - N * identity)
        o22 = 0.5 * (s2_operator @ s2_operator - N * identity)
        o01 = 0.5 * (
            s0 @ s1_operator
            + s1_operator @ s0
            - 2.0 * N * c1 * identity
        )
        o02 = 0.5 * (
            s0 @ s2_operator
            + s2_operator @ s0
            - 2.0 * N * c2 * identity
        )
        o12 = 0.5 * (
            s1_operator @ s2_operator
            + s2_operator @ s1_operator
            - 2.0 * N * np.cos(theta1 - theta2) * identity
        )
        operators = (s0, s1_operator, s2_operator, o00, o01, o02, o11, o12, o22)
        hamiltonian = sum(
            coefficient * operator
            for coefficient, operator in zip(coefficients, operators)
        )
        return hamiltonian, operators

    def energy(theta: np.ndarray) -> float:
        return min_eigpair(build(theta)[0])[0]

    def angle_gradient(theta: np.ndarray) -> np.ndarray:
        theta1, theta2 = np.asarray(theta, dtype=float)
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        hamiltonian, operators = build(theta)
        _, state = min_eigpair(hamiltonian)
        s0, s1_operator, s2_operator = operators[:3]
        ds1 = -s1 * sz + c1 * sx
        ds2 = -s2 * sz + c2 * sx

        derivative1 = (
            a1 * ds1
            + 0.5 * a11 * (ds1 @ s1_operator + s1_operator @ ds1)
            + 0.5 * a01 * (s0 @ ds1 + ds1 @ s0)
            + 0.5 * a12 * (ds1 @ s2_operator + s2_operator @ ds1)
            + a01 * N * np.sin(theta1) * identity
            + a12 * N * np.sin(theta1 - theta2) * identity
        )
        derivative2 = (
            a2 * ds2
            + 0.5 * a22 * (ds2 @ s2_operator + s2_operator @ ds2)
            + 0.5 * a02 * (s0 @ ds2 + ds2 @ s0)
            + 0.5 * a12 * (s1_operator @ ds2 + ds2 @ s1_operator)
            + a02 * N * np.sin(theta2) * identity
            - a12 * N * np.sin(theta1 - theta2) * identity
        )
        return np.asarray(
            [
                np.real(state.conj() @ (derivative1 @ state)),
                np.real(state.conj() @ (derivative2 @ state)),
            ],
            dtype=float,
        )

    grid_n = max(int(grid_n), 2)
    seeds = [
        np.asarray(
            [
                period * (i + 0.5) / grid_n,
                period * (j + 0.5) / grid_n,
            ]
        )
        for i in range(grid_n)
        for j in range(grid_n)
    ]
    seeds.extend(period * rng.random(2) for _ in range(max(0, int(rand_n))))
    seed_energies = [energy(candidate) for candidate in seeds]
    initial = seeds[int(np.argmin(seed_energies))]

    optimization = minimize(
        energy,
        initial,
        jac=angle_gradient,
        method="L-BFGS-B",
        bounds=[(0.0, period), (0.0, period)],
        options={"ftol": tol, "gtol": tol, "maxiter": 10_000},
    )
    theta = optimization.x.copy()
    hamiltonian, operators = build(theta)
    value, state = min_eigpair(hamiltonian)

    def expectation(operator: np.ndarray) -> float:
        return float(np.real(state.conj() @ (operator @ state)))

    gradient = np.asarray([expectation(operator) for operator in operators])
    theta_gradient = angle_gradient(theta)

    if report:
        print(f"[quantum N32] E0={value:.12g}")
        print(f"[quantum N32] theta={theta}")
        print(f"[quantum N32] theta gradient={theta_gradient}")
        print(f"[quantum N32] optimizer success={optimization.success}")

    return {
        "value": value,
        "theta": theta,
        "state": state,
        "grad": gradient,
        "theta_grad": theta_gradient,
        "optim": optimization,
    }


def delta_ratio(
    alpha: Sequence[float] | np.ndarray,
    N: int,
    vertices: Sequence[np.ndarray] | None = None,
    **quantum_kwargs: Any,
) -> float:
    vertices_all = enumerate_vertices_N32(N) if vertices is None else vertices
    classical = core.classical_bound(alpha, vertices_all)["bound"]
    quantum = quantum_value_grad(alpha, N, **quantum_kwargs)["value"]
    return float(quantum / classical)


def search_new_face_by_events(
    alpha_in: Sequence[float] | np.ndarray,
    N: int,
    vertices_all: Sequence[np.ndarray] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    vertices = enumerate_vertices_N32(N) if vertices_all is None else vertices_all
    return core.search_new_face_by_events(
        alpha_in,
        N,
        vertices,
        quantum_evaluator=quantum_value_grad,
        target_rank=TARGET_RANK,
        **kwargs,
    )


def run_gradient_until_facet(
    alpha_init: Sequence[float] | np.ndarray,
    N: int,
    vertices_all: Sequence[np.ndarray] | None = None,
    max_steps: int = 20,
    report: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vertices = enumerate_vertices_N32(N) if vertices_all is None else vertices_all
    return core.run_gradient_until_facet(
        alpha_init,
        N,
        vertices,
        quantum_evaluator=quantum_value_grad,
        target_rank=TARGET_RANK,
        max_steps=max_steps,
        report=report,
        **kwargs,
    )


classical_bound = core.classical_bound

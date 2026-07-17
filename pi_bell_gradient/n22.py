"""Permutation-invariant ``(N,2,2)`` Bell-gradient model."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from . import core
from .spin import collective_spin_matrices, min_eigpair


COEFFICIENT_ORDER = ("alpha0", "alpha1", "alpha00", "alpha01", "alpha11")
TARGET_RANK = len(COEFFICIENT_ORDER)


def enumerate_vertices_N22(N: int) -> list[np.ndarray]:
    """Return vertices ordered as ``[S0,S1,S00,S01,S11]``."""

    return core.enumerate_pi_vertices(N, settings=2)


def quantum_value_grad(
    alpha: Sequence[float] | np.ndarray,
    N: int,
    grid_n: int = 17,
    rand_n: int = 0,
    tol: float = 1e-10,
    seed: int | None = None,
    report: bool = False,
) -> dict[str, Any]:
    """Minimize over the relative measurement angle and return HF gradients."""

    coefficients = np.asarray(alpha, dtype=float)
    if coefficients.size != TARGET_RANK:
        raise ValueError(f"alpha must have length {TARGET_RANK}")
    a0, a1, a00, a01, a11 = coefficients

    sx, sz = collective_spin_matrices(N)
    identity = np.eye(N + 1)
    rng = np.random.default_rng(seed)

    def build(theta: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        cosine, sine = np.cos(theta), np.sin(theta)
        s0 = sz
        s1 = cosine * sz + sine * sx
        o00 = 0.5 * (s0 @ s0 - N * identity)
        o01 = 0.5 * (
            s0 @ s1 + s1 @ s0 - 2.0 * N * cosine * identity
        )
        o11 = 0.5 * (s1 @ s1 - N * identity)
        operators = (s0, s1, o00, o01, o11)
        hamiltonian = sum(
            coefficient * operator
            for coefficient, operator in zip(coefficients, operators)
        )
        return hamiltonian, operators

    def energy(x: np.ndarray) -> float:
        theta = float(np.atleast_1d(x)[0])
        return min_eigpair(build(theta)[0])[0]

    def angle_gradient_value(theta: float) -> float:
        cosine, sine = np.cos(theta), np.sin(theta)
        hamiltonian, operators = build(theta)
        _, state = min_eigpair(hamiltonian)
        s0, s1 = operators[0], operators[1]
        ds1 = -sine * sz + cosine * sx
        derivative = (
            a1 * ds1
            + 0.5 * a01 * (s0 @ ds1 + ds1 @ s0)
            + a01 * N * sine * identity
            + 0.5 * a11 * (ds1 @ s1 + s1 @ ds1)
        )
        return float(np.real(state.conj() @ (derivative @ state)))

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.asarray([angle_gradient_value(float(np.atleast_1d(x)[0]))])

    grid_n = max(int(grid_n), 3)
    seeds = [np.asarray([theta]) for theta in np.linspace(0.0, np.pi, grid_n)]
    seeds.extend(
        np.asarray([np.pi * rng.random()]) for _ in range(max(0, int(rand_n)))
    )
    seed_energies = [energy(candidate) for candidate in seeds]
    initial = seeds[int(np.argmin(seed_energies))]

    optimization = minimize(
        energy,
        initial,
        jac=jacobian,
        method="L-BFGS-B",
        bounds=[(0.0, np.pi)],
        options={"ftol": tol, "gtol": tol, "maxiter": 10_000},
    )
    theta = float(optimization.x[0])
    hamiltonian, operators = build(theta)
    value, state = min_eigpair(hamiltonian)

    def expectation(operator: np.ndarray) -> float:
        return float(np.real(state.conj() @ (operator @ state)))

    gradient = np.asarray([expectation(operator) for operator in operators])
    theta_gradient = np.asarray([angle_gradient_value(theta)])

    if report:
        print(f"[quantum N22] E0={value:.12g}")
        print(f"[quantum N22] theta={theta:.12g}")
        print(f"[quantum N22] dE/dtheta={theta_gradient[0]:.3e}")
        print(f"[quantum N22] optimizer success={optimization.success}")

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
    vertices_all = enumerate_vertices_N22(N) if vertices is None else vertices
    classical = core.classical_bound(alpha, vertices_all)["bound"]
    quantum = quantum_value_grad(alpha, N, **quantum_kwargs)["value"]
    return float(quantum / classical)


def search_new_face_by_events(
    alpha_in: Sequence[float] | np.ndarray,
    N: int,
    vertices_all: Sequence[np.ndarray] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    vertices = enumerate_vertices_N22(N) if vertices_all is None else vertices_all
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
    vertices = enumerate_vertices_N22(N) if vertices_all is None else vertices_all
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

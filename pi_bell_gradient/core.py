"""Shared polytope and constrained-gradient routines.

The search fixes the classical bound at a negative target value, projects the
Hellmann--Feynman quantum gradient onto the tangent space of the current
classical face, and moves to the first new classical-vertex event.
"""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
from itertools import product
from math import gcd
from typing import Any, Callable, Sequence

import numpy as np
from numpy.linalg import matrix_rank, norm, pinv


ArrayLike = Sequence[float] | np.ndarray
QuantumEvaluator = Callable[..., dict[str, Any]]


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def rationalize_to_integer_vector(
    values: ArrayLike,
    max_denominator: int = 10_000,
) -> tuple[int, np.ndarray, list[Fraction]]:
    """Rationally approximate a vector and return a primitive integer scale.

    The returned values satisfy ``fractions ~= integers / denominator``.
    """

    fractions = [
        Fraction(float(value)).limit_denominator(max_denominator)
        for value in np.asarray(values, dtype=float)
    ]
    denominator = 1
    for fraction in fractions:
        denominator = _lcm(denominator, fraction.denominator)

    integers = np.asarray(
        [
            fraction.numerator * (denominator // fraction.denominator)
            for fraction in fractions
        ],
        dtype=object,
    )
    common = 0
    for value in integers:
        common = gcd(common, abs(int(value)))
    if common > 1:
        integers = integers // common
        denominator //= common

    return denominator, integers, fractions


@lru_cache(maxsize=None)
def _enumerate_pi_vertices_cached(N: int, settings: int) -> tuple[tuple[int, ...], ...]:
    if N < 1:
        raise ValueError("N must be a positive integer")
    if settings < 1:
        raise ValueError("settings must be a positive integer")

    strategies = np.asarray(list(product([-1, 1], repeat=settings)), dtype=int)
    n_strategies = len(strategies)
    occupations = np.zeros(n_strategies, dtype=int)
    vertices: set[tuple[int, ...]] = set()

    cross_products = {
        (x, y): strategies[:, x] * strategies[:, y]
        for x in range(settings)
        for y in range(x + 1, settings)
    }

    def visit(k: int, remaining: int) -> None:
        if k == n_strategies - 1:
            occupations[k] = remaining
            one_body = (occupations @ strategies).astype(int)
            row = [int(value) for value in one_body]

            for x in range(settings):
                for y in range(x, settings):
                    if x == y:
                        value = (int(one_body[x]) ** 2 - N) // 2
                    else:
                        onsite = int(occupations @ cross_products[(x, y)])
                        value = int(one_body[x]) * int(one_body[y]) - onsite
                    row.append(int(value))

            vertices.add(tuple(row))
            return

        for count in range(remaining + 1):
            occupations[k] = count
            visit(k + 1, remaining - count)

    visit(0, N)
    return tuple(sorted(vertices))


def enumerate_pi_vertices(N: int, settings: int) -> list[np.ndarray]:
    """Enumerate projected PI deterministic vertices by occupation numbers.

    The coordinate order is all one-body terms followed by upper-triangular
    two-body terms.  Diagonal terms count unordered pairs, while cross terms
    follow the ordered-setting convention used in the project notebooks.
    """

    return [
        np.asarray(row, dtype=float)
        for row in _enumerate_pi_vertices_cached(int(N), int(settings))
    ]


def stack_rows(vertices: Sequence[ArrayLike]) -> np.ndarray:
    if len(vertices) == 0:
        raise ValueError("vertices is empty")
    return np.vstack([np.asarray(vertex, dtype=float) for vertex in vertices])


def classical_bound(
    alpha: ArrayLike,
    vertices: Sequence[ArrayLike],
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Return ``beta_C = min_v v @ alpha`` and its saturated vertices."""

    alpha_array = np.asarray(alpha, dtype=float)
    vertex_matrix = stack_rows(vertices)
    if vertex_matrix.shape[1] != alpha_array.size:
        raise ValueError("dimension mismatch between alpha and vertices")

    values = vertex_matrix @ alpha_array
    bound = float(np.min(values))
    active_indices = np.flatnonzero(np.abs(values - bound) <= atol)
    active_vertices = [vertex_matrix[index].copy() for index in active_indices]
    rank = (
        int(matrix_rank(stack_rows(active_vertices), tol=atol))
        if active_vertices
        else 0
    )
    return {
        "bound": bound,
        "active_vertices": active_vertices,
        "rank": rank,
    }


def same_vertex(v: ArrayLike, w: ArrayLike, atol: float = 1e-10) -> bool:
    return bool(np.allclose(v, w, atol=atol, rtol=0.0))


def merge_vertices(
    base: Sequence[ArrayLike],
    additions: Sequence[ArrayLike],
    atol: float = 1e-10,
) -> list[np.ndarray]:
    merged = [np.asarray(vertex, dtype=float).copy() for vertex in base]
    for candidate in additions:
        candidate_array = np.asarray(candidate, dtype=float)
        if not any(same_vertex(candidate_array, vertex, atol=atol) for vertex in merged):
            merged.append(candidate_array.copy())
    return merged


def new_vertices(
    base: Sequence[ArrayLike],
    candidates: Sequence[ArrayLike],
    atol: float = 1e-10,
) -> list[np.ndarray]:
    return [
        np.asarray(candidate, dtype=float).copy()
        for candidate in candidates
        if not any(same_vertex(candidate, vertex, atol=atol) for vertex in base)
    ]


def affine_project(
    alpha: ArrayLike,
    vertices: Sequence[ArrayLike],
    beta: float = -1.0,
) -> np.ndarray:
    """Project onto ``{alpha: v @ alpha = beta}`` for all given vertices."""

    active_matrix = stack_rows(vertices)
    target = np.full(active_matrix.shape[0], float(beta))
    projected = np.asarray(alpha, dtype=float).copy()
    residual = active_matrix @ projected - target
    projected -= active_matrix.T @ (pinv(active_matrix @ active_matrix.T) @ residual)
    return projected


def null_projector(vertices: Sequence[ArrayLike]) -> np.ndarray:
    """Orthogonal projector onto the null space of the active vertex rows."""

    active_matrix = stack_rows(vertices)
    dimension = active_matrix.shape[1]
    return (
        np.eye(dimension)
        - active_matrix.T
        @ pinv(active_matrix @ active_matrix.T)
        @ active_matrix
    )


def normalize_to_bound(
    alpha: ArrayLike,
    vertices_all: Sequence[ArrayLike],
    beta: float = -1.0,
    atol: float = 1e-10,
) -> tuple[np.ndarray, float, list[np.ndarray], int]:
    """Rescale a Bell vector so that its classical lower bound equals beta."""

    if beta >= 0:
        raise ValueError("the current implementation expects a negative target beta")

    result = classical_bound(alpha, vertices_all, atol=atol)
    current_bound = result["bound"]
    if current_bound >= 0:
        raise ValueError(
            "the Bell vector must have a negative classical lower bound; "
            f"received beta_C={current_bound}"
        )

    normalized = np.asarray(alpha, dtype=float) * (beta / current_bound)
    normalized_result = classical_bound(normalized, vertices_all, atol=atol)
    return (
        normalized,
        normalized_result["bound"],
        normalized_result["active_vertices"],
        normalized_result["rank"],
    )


def random_null_direction(
    projector: np.ndarray,
    scale: float = 1.0,
    rng: np.random.Generator | int | None = None,
    max_tries: int = 100,
) -> np.ndarray:
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    dimension = projector.shape[0]
    for _ in range(max_tries):
        direction = projector @ generator.standard_normal(dimension)
        direction_norm = norm(direction)
        if np.isfinite(direction_norm) and direction_norm > 1e-12:
            return scale * direction / direction_norm
    raise RuntimeError("failed to generate a nonzero null-space direction")


def candidate_events_along_direction(
    alpha0: ArrayLike,
    direction: ArrayLike,
    vertices_all: Sequence[ArrayLike],
    beta: float = -1.0,
    vertex_atol: float = 1e-10,
    denom_tol: float = 1e-12,
    t_tol: float = 1e-12,
) -> list[dict[str, Any]]:
    """Find vertex events along ``alpha(t) = alpha0 - t * direction``."""

    alpha_array = np.asarray(alpha0, dtype=float)
    direction_array = np.asarray(direction, dtype=float)
    events: list[dict[str, Any]] = []

    for vertex in vertices_all:
        vertex_array = np.asarray(vertex, dtype=float)
        slack = float(vertex_array @ alpha_array - beta)
        speed = float(vertex_array @ direction_array)
        if slack > vertex_atol and speed > denom_tol:
            event_time = slack / speed
            if event_time > t_tol and np.isfinite(event_time):
                events.append(
                    {
                        "t": event_time,
                        "vertex": vertex_array.copy(),
                        "slack": slack,
                        "speed": speed,
                    }
                )

    events.sort(key=lambda event: event["t"])
    return events


def active_vertices_at(
    alpha: ArrayLike,
    vertices_all: Sequence[ArrayLike],
    beta: float = -1.0,
    atol: float = 1e-8,
) -> tuple[list[np.ndarray], float, bool]:
    alpha_array = np.asarray(alpha, dtype=float)
    vertex_matrix = stack_rows(vertices_all)
    values = vertex_matrix @ alpha_array
    minimum = float(np.min(values))
    active_indices = np.flatnonzero(np.abs(values - beta) <= atol)
    active = [vertex_matrix[index].copy() for index in active_indices]
    return active, minimum, bool(minimum >= beta - atol)


def search_new_face_by_events(
    alpha_in: ArrayLike,
    N: int,
    vertices_all: Sequence[ArrayLike],
    quantum_evaluator: QuantumEvaluator,
    target_rank: int,
    vertices_input: Sequence[ArrayLike] | None = None,
    beta: float = -1.0,
    grad_tol: float = 1e-10,
    vertex_atol: float = 1e-10,
    active_atol: float = 1e-8,
    t_tol: float = 1e-12,
    max_events: int = 50,
    require_rank_increase: bool = True,
    rng: np.random.Generator | int | None = None,
    noise_scale: float = 1.0,
    report: bool = True,
    **quantum_kwargs: Any,
) -> dict[str, Any]:
    """Perform one event-driven constrained-gradient step."""

    alpha0, _, active0, _ = normalize_to_bound(
        alpha_in,
        vertices_all,
        beta=beta,
        atol=vertex_atol,
    )
    active_base = (
        active0
        if vertices_input is None
        else merge_vertices(active0, vertices_input, atol=vertex_atol)
    )

    alpha0 = affine_project(alpha0, active_base, beta=beta)
    active_projected, minimum_projected, valid_projected = active_vertices_at(
        alpha0,
        vertices_all,
        beta=beta,
        atol=active_atol,
    )
    if not valid_projected:
        raise ValueError(
            "the supplied active-vertex history is incompatible with the current "
            f"classical polytope: minimum={minimum_projected}, target={beta}"
        )
    active_base = merge_vertices(active_base, active_projected, atol=vertex_atol)
    rank_base = int(matrix_rank(stack_rows(active_base), tol=vertex_atol))

    classical0 = classical_bound(alpha0, vertices_all, atol=active_atol)
    beta0 = classical0["bound"]
    quantum0 = quantum_evaluator(alpha0, N, **quantum_kwargs)
    ratio0 = quantum0["value"] / beta0

    if report:
        print(
            f"[init] betaC={beta0:.12g} | active={len(active_base)} | "
            f"rank={rank_base}/{target_rank} | ratio={ratio0:.12g}"
        )

    if rank_base >= target_rank:
        return {
            "alpha": alpha0,
            "betaC": beta0,
            "quantumbound": quantum0["value"],
            "ratio": ratio0,
            "ratio_prev": ratio0,
            "ratioimprove": 0.0,
            "active_vertices": active_base,
            "vertices": active_base,
            "new_vertices": [],
            "rank": rank_base,
            "rank_base": rank_base,
            "tight_found": True,
            "direction": np.zeros_like(alpha0),
            "direction_norm": 0.0,
            "used_t": 0.0,
            "status": "already_tight",
            "events": [],
            "trace": {"alpha0": alpha0, "qres0": quantum0},
        }

    projector = null_projector(active_base)
    direction = projector @ np.asarray(quantum0["grad"], dtype=float)
    direction_norm = norm(direction)
    if (
        not np.isfinite(direction_norm)
        or direction_norm < grad_tol * max(1.0, norm(alpha0))
    ):
        direction = random_null_direction(
            projector,
            scale=noise_scale,
            rng=rng,
        )
        direction_norm = norm(direction)

    if report:
        drift = float(np.max(np.abs(stack_rows(active_base) @ direction)))
        print(f"[dir] norm={direction_norm:.6g} | old-face drift={drift:.3e}")

    events = candidate_events_along_direction(
        alpha0,
        direction,
        vertices_all,
        beta=beta,
        vertex_atol=vertex_atol,
        t_tol=t_tol,
    )
    if report:
        print(f"[events] candidates={len(events)}")

    for event_index, event in enumerate(events[:max_events], start=1):
        event_time = event["t"]
        alpha_event = alpha0 - event_time * direction
        active_event, minimum_event, valid_event = active_vertices_at(
            alpha_event,
            vertices_all,
            beta=beta,
            atol=active_atol,
        )
        added = new_vertices(active_base, active_event, atol=vertex_atol)
        merged = merge_vertices(active_base, active_event, atol=vertex_atol)
        rank_event = int(matrix_rank(stack_rows(merged), tol=vertex_atol))
        accept = (
            valid_event
            and bool(added)
            and (not require_rank_increase or rank_event > rank_base)
        )

        if report:
            print(
                f"[check] k={event_index} | t={event_time:.12g} | "
                f"min={minimum_event:.12g} | new={len(added)} | "
                f"rank={rank_event}/{target_rank} | accept={accept}"
            )

        if accept:
            classical_event = classical_bound(alpha_event, vertices_all, atol=active_atol)
            quantum_event = quantum_evaluator(alpha_event, N, **quantum_kwargs)
            ratio_event = quantum_event["value"] / classical_event["bound"]
            return {
                "alpha": alpha_event,
                "betaC": classical_event["bound"],
                "quantumbound": quantum_event["value"],
                "ratio": ratio_event,
                "ratio_prev": ratio0,
                "ratioimprove": ratio_event - ratio0,
                "active_vertices": active_event,
                "vertices": merged,
                "new_vertices": added,
                "rank": rank_event,
                "rank_base": rank_base,
                "tight_found": rank_event >= target_rank,
                "direction": direction,
                "direction_norm": direction_norm,
                "used_t": event_time,
                "status": "new_face_found",
                "events": events,
                "trace": {
                    "alpha0": alpha0,
                    "active0": active0,
                    "active_base": active_base,
                    "qres0": quantum0,
                    "qres_final": quantum_event,
                },
            }

    status = "no_event" if not events else "no_valid_event"
    return {
        "alpha": alpha0,
        "betaC": beta0,
        "quantumbound": quantum0["value"],
        "ratio": ratio0,
        "ratio_prev": ratio0,
        "ratioimprove": 0.0,
        "active_vertices": active_base,
        "vertices": active_base,
        "new_vertices": [],
        "rank": rank_base,
        "rank_base": rank_base,
        "tight_found": rank_base >= target_rank,
        "direction": direction,
        "direction_norm": direction_norm,
        "used_t": 0.0,
        "status": status,
        "events": events,
        "trace": {"alpha0": alpha0, "active0": active0, "qres0": quantum0},
    }


def run_gradient_until_facet(
    alpha_init: ArrayLike,
    N: int,
    vertices_all: Sequence[ArrayLike],
    quantum_evaluator: QuantumEvaluator,
    target_rank: int,
    max_steps: int = 20,
    report: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Repeat event-driven steps until full active-row rank or a stop event."""

    alpha = np.asarray(alpha_init, dtype=float)
    history: list[dict[str, Any]] = []
    active_history: Sequence[ArrayLike] | None = None

    for step in range(1, max_steps + 1):
        if report:
            print(f"Gradient step {step}, N={N}")

        result = search_new_face_by_events(
            alpha,
            N,
            vertices_all,
            quantum_evaluator=quantum_evaluator,
            target_rank=target_rank,
            vertices_input=active_history,
            report=report,
            **kwargs,
        )
        history.append(result)
        alpha = result["alpha"]
        active_history = result["vertices"]

        if report:
            print(
                f"[summary] status={result['status']} | "
                f"rank={result['rank']}/{target_rank} | "
                f"ratio={result['ratio']:.12g}"
            )

        if result["tight_found"] or result["status"] != "new_face_found":
            break

    return alpha, history

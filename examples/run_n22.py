"""Small ``(N,2,2)`` constrained-gradient example."""

from __future__ import annotations

import argparse

import numpy as np

from pi_bell_gradient import n22


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--grid", type=int, default=17)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # A small deterministic perturbation makes the starting inequality
    # non-tight, so the example visibly exercises the event-driven search.
    alpha_init = np.asarray([60.0, 32.0, 36.0, 24.0, 5.0])
    alpha_init += 1e-3 * np.arange(1, n22.TARGET_RANK + 1)
    vertices = n22.enumerate_vertices_N22(args.N)
    alpha, history = n22.run_gradient_until_facet(
        alpha_init,
        args.N,
        vertices_all=vertices,
        max_steps=args.steps,
        report=not args.quiet,
        grid_n=args.grid,
        rand_n=0,
        seed=args.seed,
        rng=args.seed,
    )
    final = history[-1]
    print("scenario = N22")
    print(f"N = {args.N}")
    print(f"vertices = {len(vertices)}")
    print(f"status = {final['status']}")
    print(f"rank = {final['rank']}/{n22.TARGET_RANK}")
    print(f"betaC = {final['betaC']:.12g}")
    print(f"ratio = {final['ratio']:.12g}")
    print(f"alpha = {np.array2string(alpha, precision=10)}")


if __name__ == "__main__":
    main()

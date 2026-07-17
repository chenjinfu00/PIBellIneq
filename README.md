# Constrained gradient optimization for PI Bell inequalities

This repository contains a standalone implementation of the current
coefficient-space gradient search for permutation-invariant Bell inequalities
in the `(N,2,2)` (N22) and `(N,3,2)` (N32) scenarios.

The public code is intentionally smaller than the research notebooks: it keeps
vertex enumeration, exact finite-`N` classical bounds, symmetric-sector quantum
optimization, analytic Hellmann--Feynman gradients, and the event-driven search.
SDP/cdd experiments, plots, stored scan results, and result-specific scratch
cells remain outside this folder.

## Method

For a Bell coefficient vector `alpha`, the code uses

```text
beta_C(alpha) = min_v v dot alpha,
beta_Q(alpha) = min_theta lambda_min(H(alpha, theta)),
Delta(alpha)  = beta_Q(alpha) / beta_C(alpha).
```

The search assumes the lower-bound convention `beta_C < 0` and rescales each
iterate to `beta_C = -1`. If the rows of `A` are the currently saturated
classical vertices, admissible coefficient directions lie in `null(A)`. The
projected Hellmann--Feynman direction is

```text
P = I - A.T (A A.T)^+ A,
d = P grad_alpha beta_Q.
```

The code follows `alpha(t) = alpha - t d` to the first event at which a new
classical vertex saturates the bound. Repeating this step grows the active-row
rank until a facet candidate is reached or no valid event remains.

This is a numerical search, not a proof of global quantum optimality. The
quantum minimization is restricted to real coplanar measurements and the
permutation-symmetric spin-`N/2` sector. Degenerate ground states can also make
a single-state Hellmann--Feynman direction nonsmooth.

## Coordinate conventions

N22 uses

```text
[alpha0, alpha1, alpha00, alpha01, alpha11]
[S0,     S1,     S00,     S01,     S11]
```

N32 uses

```text
[alpha0, alpha1, alpha2, alpha00, alpha01, alpha02, alpha11, alpha12, alpha22]
[S0,     S1,     S2,     S00,     S01,     S02,     S11,     S12,     S22]
```

For deterministic outcomes `a_x^(i) in {+1,-1}`,

```text
Sx  = sum_i a_x^(i),
Sxx = (Sx^2 - N)/2,
Sxy = Sx Sy - sum_i a_x^(i) a_y^(i)  (x != y).
```

The quantum collective operators use the same Pauli-sum normalization, with
spectrum `-N,-N+2,...,N`.

## Installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Only NumPy and SciPy are required.

## Examples

```bash
python -m examples.run_n22 --N 6 --steps 5
python -m examples.run_n32 --N 5 --steps 10
```

The N32 deterministic-vertex enumeration scales as the number of compositions
of `N` into eight single-site strategies, so large `N` can become the dominant
cost.

## Layout

```text
pi_bell_gradient/core.py   shared vertices and event-driven search
pi_bell_gradient/spin.py   collective Pauli-sum matrices
pi_bell_gradient/n22.py    N22 Bell operator and public API
pi_bell_gradient/n32.py    N32 Bell operator and public API
examples/                  command-line examples
```

An open-source license and citation metadata have not yet been selected. Add
the authors' preferred license and citation information before a formal
software release.

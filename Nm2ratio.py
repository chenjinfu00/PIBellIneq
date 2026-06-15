from itertools import product
from fractions import Fraction
from math import comb
import time
import csv
import numpy as np
from functools import lru_cache
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize, differential_evolution


def to_fraction_list(v):
    """
    Convert a list/tuple of ints, floats-as-strings, or Fractions to Fractions.

    Recommended:
        use integers or Fraction objects.
        For rational numbers, use Fraction(1, 3), not 1/3.
    """
    return [x if isinstance(x, Fraction) else Fraction(x) for x in v]


def compress_strategies_dp(alpha, gamma):
    """
    First-layer DP.

    Compress all 2^m deterministic strategies into attainable values of

        u = gamma · a

    and keep the minimal value of

        alpha · a

    for each u.

    Parameters
    ----------
    alpha, gamma : list
        m-dimensional vectors. Entries can be int or Fraction.

    Returns
    -------
    cost : dict
        u -> DP_str_m(u) - u^2 / 2

    hmin : dict
        u -> DP_str_m(u), i.e. minimal alpha · a for fixed gamma · a = u.
    """
    alpha = to_fraction_list(alpha)
    gamma = to_fraction_list(gamma)

    if len(alpha) != len(gamma):
        raise ValueError("alpha and gamma must have the same length.")

    # DP_str_0(0) = 0
    dp = {Fraction(0): Fraction(0)}

    for aj, gj in zip(alpha, gamma):
        new_dp = {}

        for u, h in dp.items():
            # Choose a_j = +1
            u_plus = u + gj
            h_plus = h + aj

            if u_plus not in new_dp or h_plus < new_dp[u_plus]:
                new_dp[u_plus] = h_plus

            # Choose a_j = -1
            u_minus = u - gj
            h_minus = h - aj

            if u_minus not in new_dp or h_minus < new_dp[u_minus]:
                new_dp[u_minus] = h_minus

        dp = new_dp

    hmin = dp
    cost = {
        u: h - Fraction(1, 2) * u * u
        for u, h in hmin.items()
    }

    return cost, hmin


def classical_bounds_rank1_dp(alpha, gamma, Nmax, keep_argmin=False):
    """
    Exact finite-N classical bound for rank-one Bell inequalities.

    This computes beta_C(N) for all N = 1, ..., Nmax in one run.

    Parameters
    ----------
    alpha : list
        m-dimensional vector alpha.

    gamma : list
        m-dimensional vector gamma.

    Nmax : int
        Compute bounds for all N = 1, ..., Nmax.

    keep_argmin : bool
        If True, also stores the optimal total w for each N.

    Returns
    -------
    result : dict
        {
            "bounds": dict, N -> beta_C(N),
            "compressed_cost": dict, u -> minimal single-site cost,
            "hmin": dict, u -> minimal alpha · a,
            "last_dp": dict, final particle-level DP state,
            "argmins": dict or None, N -> optimal w,
        }
    """
    cost, hmin = compress_strategies_dp(alpha, gamma)

    # DP^N_0(0) = 0
    dp = {Fraction(0): Fraction(0)}

    bounds = {}
    argmins = {} if keep_argmin else None

    for r in range(1, Nmax + 1):
        new_dp = {}

        for w, q in dp.items():
            for u, cu in cost.items():
                w_new = w + u
                q_new = q + cu

                if w_new not in new_dp or q_new < new_dp[w_new]:
                    new_dp[w_new] = q_new

        dp = new_dp

        best_w, best_value = min(
            (
                (w, Fraction(1, 2) * w * w + q)
                for w, q in dp.items()
            ),
            key=lambda x: x[1],
        )

        bounds[r] = best_value

        if keep_argmin:
            argmins[r] = best_w

    return {
        "bounds": bounds,
        "compressed_cost": cost,
        "hmin": hmin,
        "last_dp": dp,
        "argmins": argmins,
    }


def direct_bound_rank1(alpha, gamma, N):
    """
    Direct enumeration over all occupation numbers.

    This is only for checking small m and small N, because it becomes expensive.

    Parameters
    ----------
    alpha, gamma : list
        m-dimensional vectors.

    N : int
        Number of particles.

    Returns
    -------
    dict
        {
            "bound": exact classical bound,
            "occupations": optimal occupation vector,
            "strategies": list of strategies,
        }
    """
    alpha = to_fraction_list(alpha)
    gamma = to_fraction_list(gamma)

    m = len(alpha)
    strategies = list(product([-1, 1], repeat=m))

    items = []

    for a in strategies:
        u = sum(gamma[i] * a[i] for i in range(m))
        h = sum(alpha[i] * a[i] for i in range(m))
        c = h - Fraction(1, 2) * u * u
        items.append((u, c, a))

    K = len(items)

    best = None
    best_occ = None
    occ = [0] * K

    def rec(idx, remaining):
        nonlocal best, best_occ

        if idx == K - 1:
            occ[idx] = remaining

            w = sum(occ[i] * items[i][0] for i in range(K))
            q = sum(occ[i] * items[i][1] for i in range(K))
            value = Fraction(1, 2) * w * w + q

            if best is None or value < best:
                best = value
                best_occ = occ.copy()

            return

        for n in range(remaining + 1):
            occ[idx] = n
            rec(idx + 1, remaining - n)

    rec(0, N)

    return {
        "bound": best,
        "occupations": best_occ,
        "strategies": [item[2] for item in items],
    }


def run_test_case(alpha, gamma, N, do_direct=True):
    """
    Run one test case and optionally compare DP with direct enumeration.
    """
    m = len(alpha)

    t0 = time.perf_counter()
    dp_result = classical_bounds_rank1_dp(alpha, gamma, N, keep_argmin=True)
    t1 = time.perf_counter()

    dp_bound = dp_result["bounds"][N]
    dp_time = t1 - t0

    compressed_states = len(dp_result["compressed_cost"])
    final_w_states = len(dp_result["last_dp"])

    print("=" * 80)
    print(f"m = {m}, N = {N}")
    print("alpha =", alpha)
    print("gamma =", gamma)
    print("DP bound =", dp_bound, "≈", float(dp_bound))
    print("optimal w =", dp_result["argmins"][N])
    print("compressed u states =", compressed_states)
    print("final w states =", final_w_states)
    print("DP time =", dp_time, "seconds")

    if do_direct:
        candidates = comb(N + 2**m - 1, 2**m - 1)

        t2 = time.perf_counter()
        direct_result = direct_bound_rank1(alpha, gamma, N)
        t3 = time.perf_counter()

        direct_bound = direct_result["bound"]
        direct_time = t3 - t2

        print("direct bound =", direct_bound, "≈", float(direct_bound))
        print("direct candidates =", candidates)
        print("direct time =", direct_time, "seconds")
        print("same bound?", dp_bound == direct_bound)

    return dp_result









# ---------------------------------------------------------------------
# Spin data in the symmetric subspace
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def spin_tridiagonal_data(N):
    """
    Spin-S=N/2 data in the Sz basis.

    Basis:
        |S,m>, m = S, S-1, ..., -S.

    Returns
    -------
    mvals : ndarray
        Diagonal entries of Sz.

    sx_off : ndarray
        Off-diagonal entries of Sx.
        Sx has sx_off[i] between basis states i and i+1.
    """
    S = N / 2
    dim = N + 1

    mvals = np.array([S - i for i in range(dim)], dtype=float)

    sx_off = np.zeros(dim - 1, dtype=float)

    for i in range(dim - 1):
        m = mvals[i]
        sx_off[i] = 0.5 * np.sqrt(S * (S + 1) - m * (m - 1))

    return mvals, sx_off


# ---------------------------------------------------------------------
# Bell coefficients
# ---------------------------------------------------------------------

def bell_coefficients_rank1(alpha, gamma, theta, N):
    """
    Compute

        J, Jx, Jz, gamma_x, gamma_z

    for the rank-one Bell operator

        I(theta)
        =
        J
        + Jx Sx
        + Jz Sz
        + 1/2 (gamma_x Sx + gamma_z Sz)^2.
    """
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if not (len(alpha) == len(gamma) == len(theta)):
        raise ValueError("alpha, gamma, and theta must have the same length.")

    Jx = 2 * np.sum(alpha * np.sin(theta))
    Jz = 2 * np.sum(alpha * np.cos(theta))

    gamma_x = 2 * np.sum(gamma * np.sin(theta))
    gamma_z = 2 * np.sum(gamma * np.cos(theta))

    J = -N / 8 * (gamma_x**2 + gamma_z**2)

    return {
        "J": J,
        "Jx": Jx,
        "Jz": Jz,
        "gamma_x": gamma_x,
        "gamma_z": gamma_z,
    }


def rotated_coefficients_rank1(alpha, gamma, theta, N):
    """
    Rotate the quadratic direction

        gamma_x Sx + gamma_z Sz

    to the new z-axis.

    In the rotated basis,

        I =
        J
        + h_perp Sx
        + h_parallel Sz
        + 1/2 rho^2 Sz^2.

    This matrix is tridiagonal in the Sz basis.
    """
    coeffs = bell_coefficients_rank1(alpha, gamma, theta, N)

    Jx = coeffs["Jx"]
    Jz = coeffs["Jz"]
    gamma_x = coeffs["gamma_x"]
    gamma_z = coeffs["gamma_z"]

    rho = np.hypot(gamma_x, gamma_z)

    if rho < 1e-14:
        h_parallel = Jz
        h_perp = Jx
    else:
        h_parallel = (Jx * gamma_x + Jz * gamma_z) / rho
        h_perp = (Jx * gamma_z - Jz * gamma_x) / rho

    coeffs.update({
        "rho": rho,
        "h_parallel": h_parallel,
        "h_perp": h_perp,
    })

    return coeffs


# ---------------------------------------------------------------------
# Tridiagonal Bell operator and ground-state energy
# ---------------------------------------------------------------------

def bell_tridiagonal_rank1(alpha, gamma, theta, N):
    """
    Return the diagonal and off-diagonal entries of the tridiagonal Bell operator.

    In the rotated basis:

        I =
        J
        + h_perp Sx
        + h_parallel Sz
        + 1/2 rho^2 Sz^2.
    """
    coeffs = rotated_coefficients_rank1(alpha, gamma, theta, N)
    mvals, sx_off = spin_tridiagonal_data(N)

    J = coeffs["J"]
    h_parallel = coeffs["h_parallel"]
    h_perp = coeffs["h_perp"]
    rho = coeffs["rho"]

    diag = (
        J
        + h_parallel * mvals
        + 0.5 * rho**2 * mvals**2
    )

    offdiag = h_perp * sx_off

    return diag, offdiag, coeffs


def quantum_value_rank1_tridiagonal(alpha, gamma, theta, N, return_eigenvector=False):
    """
    Fast exact ground-state energy using tridiagonal diagonalization.
    """
    diag, offdiag, coeffs = bell_tridiagonal_rank1(alpha, gamma, theta, N)

    if return_eigenvector:
        eigvals, eigvecs = eigh_tridiagonal(
            diag,
            offdiag,
            select="i",
            select_range=(0, 0),
        )

        return {
            "beta_Q": eigvals[0],
            "ground_state": eigvecs[:, 0],
            "coeffs": coeffs,
        }

    eigvals = eigh_tridiagonal(
        diag,
        offdiag,
        select="i",
        select_range=(0, 0),
        eigvals_only=True,
    )

    return {
        "beta_Q": eigvals[0],
        "coeffs": coeffs,
    }


# ---------------------------------------------------------------------
# Reverse-symmetric measurement parametrization
# ---------------------------------------------------------------------

def theta_from_reverse_params(params, m):
    """
    Build full theta vector from reverse-symmetry parameters.

    Constraint:
        theta_k = -theta_{m-1-k}.

    For even m:
        params = [theta_0, ..., theta_{m/2-1}]
        theta  = [theta_0, ..., theta_{m/2-1},
                  -theta_{m/2-1}, ..., -theta_0]

    For odd m:
        params = [theta_0, ..., theta_{(m-3)/2}]
        theta  = [theta_0, ..., theta_{(m-3)/2},
                  0,
                  -theta_{(m-3)/2}, ..., -theta_0]
    """
    params = np.asarray(params, dtype=float)
    half = m // 2

    if len(params) != half:
        raise ValueError(f"Expected {half} parameters for m={m}, got {len(params)}.")

    if m % 2 == 0:
        theta = np.concatenate([params, -params[::-1]])
    else:
        theta = np.concatenate([params, np.array([0.0]), -params[::-1]])

    return theta


# ---------------------------------------------------------------------
# Hellmann--Feynman gradient in tridiagonal representation
# ---------------------------------------------------------------------

def quantum_value_and_grad_full_theta_tridiagonal(alpha, gamma, theta, N):
    """
    Ground-state energy and gradient with respect to full theta.

    Uses the tridiagonal representation and Hellmann--Feynman theorem.
    """
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    theta = np.asarray(theta, dtype=float)

    diag, offdiag, coeffs = bell_tridiagonal_rank1(alpha, gamma, theta, N)

    eigvals, eigvecs = eigh_tridiagonal(
        diag,
        offdiag,
        select="i",
        select_range=(0, 0),
    )

    beta = eigvals[0]
    psi = eigvecs[:, 0]

    mvals, sx_off = spin_tridiagonal_data(N)

    prob = psi**2

    exp_Sz = np.sum(prob * mvals)
    exp_Sz2 = np.sum(prob * mvals**2)

    # Since Sx is tridiagonal:
    # <Sx> = 2 sum_i sx_off[i] psi_i psi_{i+1}
    exp_Sx = 2.0 * np.sum(sx_off * psi[:-1] * psi[1:])

    Jx = coeffs["Jx"]
    Jz = coeffs["Jz"]
    gamma_x = coeffs["gamma_x"]
    gamma_z = coeffs["gamma_z"]
    rho = coeffs["rho"]

    grad = np.zeros_like(theta, dtype=float)

    for a in range(len(theta)):
        dJx = 2 * alpha[a] * np.cos(theta[a])
        dJz = -2 * alpha[a] * np.sin(theta[a])

        dgamma_x = 2 * gamma[a] * np.cos(theta[a])
        dgamma_z = -2 * gamma[a] * np.sin(theta[a])

        dJ = -N / 4 * (
            gamma_x * dgamma_x
            + gamma_z * dgamma_z
        )

        if rho < 1e-14:
            dh_parallel = dJz
            dh_perp = dJx
            d_rho2 = 0.0
        else:
            q = gamma_x * dgamma_x + gamma_z * dgamma_z
            drho = q / rho
            d_rho2 = 2 * q

            # h_parallel = (Jx gamma_x + Jz gamma_z) / rho
            num_parallel = Jx * gamma_x + Jz * gamma_z
            dnum_parallel = (
                dJx * gamma_x
                + Jx * dgamma_x
                + dJz * gamma_z
                + Jz * dgamma_z
            )

            dh_parallel = (
                dnum_parallel * rho
                - num_parallel * drho
            ) / rho**2

            # h_perp = (Jx gamma_z - Jz gamma_x) / rho
            num_perp = Jx * gamma_z - Jz * gamma_x
            dnum_perp = (
                dJx * gamma_z
                + Jx * dgamma_z
                - dJz * gamma_x
                - Jz * dgamma_x
            )

            dh_perp = (
                dnum_perp * rho
                - num_perp * drho
            ) / rho**2

        grad[a] = (
            dJ
            + dh_parallel * exp_Sz
            + dh_perp * exp_Sx
            + 0.5 * d_rho2 * exp_Sz2
        )

    return beta, grad, coeffs


def quantum_value_and_grad_reverse_params_tridiagonal(params, alpha, gamma, N):
    """
    beta_Q and gradient with respect to reverse-symmetric parameters.
    """
    m = len(alpha)
    theta = theta_from_reverse_params(params, m)

    beta, grad_theta, coeffs = quantum_value_and_grad_full_theta_tridiagonal(
        alpha=alpha,
        gamma=gamma,
        theta=theta,
        N=N,
    )

    half = m // 2
    grad_params = np.zeros(half, dtype=float)

    for r in range(half):
        grad_params[r] = grad_theta[r] - grad_theta[m - 1 - r]

    return beta, grad_params, coeffs


# ---------------------------------------------------------------------
# Gradient-based optimization
# ---------------------------------------------------------------------

def optimize_theta_reverse_symmetry_gradient_fast(
    alpha,
    gamma,
    N,
    theta0_params=None,
    method="BFGS",
    maxiter=1000,
    gtol=1e-8,
    verbose=True,
):
    """
    Optimize beta_Q(theta) under reverse symmetry using tridiagonal solver
    and analytic Hellmann--Feynman gradient.
    """
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    if len(alpha) != len(gamma):
        raise ValueError("alpha and gamma must have the same length.")

    m = len(alpha)
    half = m // 2

    if theta0_params is None:
        theta0_params = np.linspace(0.6, 0.2, half)
    else:
        theta0_params = np.asarray(theta0_params, dtype=float)

    def fun(params):
        beta, grad, coeffs = quantum_value_and_grad_reverse_params_tridiagonal(
            params=params,
            alpha=alpha,
            gamma=gamma,
            N=N,
        )
        return beta

    def jac(params):
        beta, grad, coeffs = quantum_value_and_grad_reverse_params_tridiagonal(
            params=params,
            alpha=alpha,
            gamma=gamma,
            N=N,
        )
        return grad

    opt = minimize(
        fun,
        theta0_params,
        jac=jac,
        method=method,
        options={
            "maxiter": maxiter,
            "gtol": gtol,
        },
    )

    theta_full = theta_from_reverse_params(opt.x, m)

    beta, grad, coeffs = quantum_value_and_grad_reverse_params_tridiagonal(
        params=opt.x,
        alpha=alpha,
        gamma=gamma,
        N=N,
    )

    if verbose:
        print("=" * 80)
        print(f"Fast tridiagonal gradient optimization: m={m}, N={N}")
        print("alpha =", alpha)
        print("gamma =", gamma)
        print("success =", opt.success)
        print("message =", opt.message)
        print("beta_Q =", beta)
        print("theta params =", opt.x)
        print("theta full =", theta_full)
        print("gradient =", grad)
        print("gradient norm =", np.linalg.norm(grad))
        print("coeffs =", coeffs)
        print("nfev =", opt.nfev)
        print("njev =", opt.njev)

    return {
        "success": opt.success,
        "message": opt.message,
        "beta_Q": beta,
        "theta_params": opt.x,
        "theta_full": theta_full,
        "grad": grad,
        "grad_norm": np.linalg.norm(grad),
        "coeffs": coeffs,
        "optimizer_result": opt,
    }


# ---------------------------------------------------------------------
# Ratio calculation
# ---------------------------------------------------------------------

def compute_ratio_rank1_fast(
    alpha,
    gamma,
    N,
    classical_slope,
    theta0_params=None,
    method="BFGS",
    maxiter=1000,
    gtol=1e-8,
    verbose=True,
):
    """
    Compute

        beta_Q(N), beta_C(N), ratio = beta_Q / beta_C

    using fast tridiagonal gradient optimization.

    Assumes

        beta_C(N) = classical_slope * N.
    """
    qres = optimize_theta_reverse_symmetry_gradient_fast(
        alpha=alpha,
        gamma=gamma,
        N=N,
        theta0_params=theta0_params,
        method=method,
        maxiter=maxiter,
        gtol=gtol,
        verbose=verbose,
    )

    beta_Q = qres["beta_Q"]
    beta_C = classical_slope * N
    ratio = beta_Q / beta_C

    result = {
        "N": N,
        "beta_Q": beta_Q,
        "beta_C": beta_C,
        "ratio": ratio,
        "theta_params": qres["theta_params"],
        "theta_full": qres["theta_full"],
        "grad_norm": qres["grad_norm"],
        "success": qres["success"],
        "message": qres["message"],
        "coeffs": qres["coeffs"],
        "optimizer_result": qres["optimizer_result"],
    }

    if verbose:
        print("-" * 80)
        print("beta_C =", beta_C)
        print("ratio beta_Q / beta_C =", ratio)

    return result


def scan_N_warm_start_rank1_fast(
    alpha,
    gamma,
    classical_slope,
    N_min,
    N_max,
    theta0_params=None,
    method="BFGS",
    maxiter=1000,
    gtol=1e-8,
    verbose=True,
):
    """
    Scan N using warm start.

    For each N, the optimal theta_params from the previous N
    are used as the initial point for the next N.
    """
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    m = len(alpha)
    half = m // 2

    if theta0_params is None:
        theta0_params = np.linspace(0.6, 0.2, half)
    else:
        theta0_params = np.asarray(theta0_params, dtype=float)

    current_theta0 = theta0_params.copy()
    results = []

    for N in range(N_min, N_max + 1):
        row = compute_ratio_rank1_fast(
            alpha=alpha,
            gamma=gamma,
            N=N,
            classical_slope=classical_slope,
            theta0_params=current_theta0,
            method=method,
            maxiter=maxiter,
            gtol=gtol,
            verbose=False,
        )

        current_theta0 = row["theta_params"].copy()
        results.append(row)

        if verbose:
            print(
                f"N={N:5d}, "
                f"beta_Q={row['beta_Q']:.12f}, "
                f"beta_C={row['beta_C']:.12f}, "
                f"ratio={row['ratio']:.12f}, "
                f"theta={row['theta_full']}, "
                f"grad_norm={row['grad_norm']:.3e}"
            )

    return results


def save_scan_results_csv(results, filename):
    """
    Save scan results to CSV.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "N",
                "beta_Q",
                "beta_C",
                "ratio",
                "theta_params",
                "theta_full",
                "grad_norm",
                "success",
                "message",
            ],
        )
        writer.writeheader()

        for row in results:
            writer.writerow({
                "N": row["N"],
                "beta_Q": row["beta_Q"],
                "beta_C": row["beta_C"],
                "ratio": row["ratio"],
                "theta_params": row["theta_params"].tolist(),
                "theta_full": row["theta_full"].tolist(),
                "grad_norm": row["grad_norm"],
                "success": row["success"],
                "message": row["message"],
            })







def wrap_phi(phi):
    """
    Wrap angle to [-pi, pi].
    """
    return np.arctan2(np.sin(phi), np.cos(phi))


# ============================================================
# Bell coefficients
# ============================================================

def bell_coefficients_rank1(alpha, gamma, theta, N):
    """
    Compute coefficients for

        I(theta)
        =
        J + Jx Sx + Jz Sz
        + 1/2 (gamma_x Sx + gamma_z Sz)^2.
    """
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if not (len(alpha) == len(gamma) == len(theta)):
        raise ValueError("alpha, gamma, and theta must have the same length.")

    Jx = 2.0 * np.sum(alpha * np.sin(theta))
    Jz = 2.0 * np.sum(alpha * np.cos(theta))

    gamma_x = 2.0 * np.sum(gamma * np.sin(theta))
    gamma_z = 2.0 * np.sum(gamma * np.cos(theta))

    J = -N / 8.0 * (gamma_x**2 + gamma_z**2)

    return {
        "J": J,
        "Jx": Jx,
        "Jz": Jz,
        "gamma_x": gamma_x,
        "gamma_z": gamma_z,
    }


# ============================================================
# HP approximation
# ============================================================

def hp_ground_energy_from_coeffs(coeffs, N, use_abs_Jperp=True):
    """
    HP approximate ground-state energy.

    E_HP =
        J - K_perp S
        - K_perp/2
        + 1/2 sqrt[K_perp (K_perp + gamma^2 S)]
        - J_gamma^2 S / [2 (K_perp + gamma^2 S)].

    If use_abs_Jperp=True, K_perp=abs(J_perp_raw).
    """
    S = N / 2.0

    J = coeffs["J"]
    Jx = coeffs["Jx"]
    Jz = coeffs["Jz"]
    gamma_x = coeffs["gamma_x"]
    gamma_z = coeffs["gamma_z"]

    gamma2 = gamma_x**2 + gamma_z**2
    gamma_norm = np.sqrt(gamma2)

    if gamma_norm < 1e-14:
        raise ValueError("gamma_x and gamma_z are both close to zero.")

    J_gamma = (Jx * gamma_x + Jz * gamma_z) / gamma_norm
    J_perp_raw = (Jx * gamma_z - Jz * gamma_x) / gamma_norm

    K_perp = abs(J_perp_raw) if use_abs_Jperp else J_perp_raw

    denom = K_perp + gamma2 * S

    if K_perp <= 0 or denom <= 0:
        E_HP = np.nan
    else:
        E_HP = (
            J
            - K_perp * S
            - K_perp / 2.0
            + 0.5 * np.sqrt(K_perp * denom)
            - (J_gamma**2 * S) / (2.0 * denom)
        )

    return {
        "E_HP": E_HP,
        "gamma2": gamma2,
        "J_gamma": J_gamma,
        "J_perp_raw": J_perp_raw,
        "K_perp": K_perp,
        "HP_denom": denom,
    }


def hp_ground_energy_rank1(alpha, gamma, theta, N, use_abs_Jperp=True):
    """
    Compute HP energy from alpha, gamma, theta, N.
    """
    coeffs = bell_coefficients_rank1(
        alpha=alpha,
        gamma=gamma,
        theta=theta,
        N=N,
    )

    hp_res = hp_ground_energy_from_coeffs(
        coeffs=coeffs,
        N=N,
        use_abs_Jperp=use_abs_Jperp,
    )

    return {
        **hp_res,
        "coeffs": coeffs,
    }


# ============================================================
# Stable certified moment objective
# ============================================================

def certified_f_rho_phi_fixed_theta_stable(rho, phi, coeffs, N):
    """
    Stable certified moment objective.

    Variables:
        rho = r/S in [0,1],
        phi.

    Parametrization:
        Sx = r sin(phi),
        Sz = r cos(phi),
        r = rho S,
        S = N/2.

    The covariance bracket is evaluated as

        T - sqrt(T^2-r^2)
        =
        r^2 / (T + sqrt(T^2-r^2)),

    which is numerically stable for large N.
    """
    S = N / 2.0

    if rho < 0.0 or rho > 1.0:
        return np.inf

    phi = wrap_phi(phi)
    r = rho * S

    J = coeffs["J"]
    Jx = coeffs["Jx"]
    Jz = coeffs["Jz"]
    gamma_x = coeffs["gamma_x"]
    gamma_z = coeffs["gamma_z"]

    Sx = r * np.sin(phi)
    Sz = r * np.cos(phi)

    T = S * (S + 1.0) - r**2
    rad = T**2 - r**2

    if rad < -1e-8:
        return np.inf

    rad = max(rad, 0.0)
    sqrt_rad = np.sqrt(rad)

    if abs(r) < 1e-15:
        bracket = 0.0
    else:
        bracket = r**2 / (T + sqrt_rad)

    transverse_gamma = gamma_x * np.cos(phi) - gamma_z * np.sin(phi)

    value = (
        J
        + Jx * Sx
        + Jz * Sz
        + 0.5 * (gamma_x * Sx + gamma_z * Sz) ** 2
        + 0.25 * transverse_gamma**2 * bracket
    )

    return value


# ============================================================
# Initial guesses
# ============================================================

def deterministic_certified_guesses(coeffs):
    """
    Initial guesses used only for the first point, or for global fallback.
    """
    guesses = []

    # Generic guesses near boundary and several directions.
    for rho in [0.999999, 0.999, 0.99, 0.95, 0.9, 0.75]:
        for phi in [0.0, np.pi / 2, -np.pi / 2, np.pi]:
            guesses.append([rho, phi])

    # Transverse-to-gamma guess.
    Jx = coeffs["Jx"]
    Jz = coeffs["Jz"]
    gx = coeffs["gamma_x"]
    gz = coeffs["gamma_z"]

    gnorm = np.hypot(gx, gz)

    if gnorm > 1e-14:
        perp = np.array([gz, -gx]) / gnorm

        if Jx * perp[0] + Jz * perp[1] > 0:
            perp = -perp

        phi_perp = np.arctan2(perp[0], perp[1])

        for rho in [0.999999, 0.999, 0.99, 0.95]:
            guesses.append([rho, phi_perp])

    return guesses


def warm_certified_guesses(x0):
    """
    Very small set of guesses around previous optimum.
    This is the key speedup.
    """
    if x0 is None:
        return []

    x0 = np.asarray(x0, dtype=float).copy()
    rho0 = np.clip(x0[0], 0.0, 1.0)
    phi0 = wrap_phi(x0[1])

    guesses = [
        [rho0, phi0],
        [np.clip(rho0 - 1e-6, 0.0, 1.0), phi0],
        [np.clip(rho0 + 1e-6, 0.0, 1.0), phi0],
        [np.clip(rho0 - 1e-5, 0.0, 1.0), phi0],
        [np.clip(rho0 + 1e-5, 0.0, 1.0), phi0],
        [rho0, wrap_phi(phi0 + 1e-6)],
        [rho0, wrap_phi(phi0 - 1e-6)],
        [rho0, wrap_phi(phi0 + 1e-5)],
        [rho0, wrap_phi(phi0 - 1e-5)],
    ]

    return guesses


# ============================================================
# Fast warm-start certified optimizer
# ============================================================

def optimize_certified_fixed_theta_fast_warm(
    alpha,
    gamma,
    theta,
    N,
    classical_slope=None,
    x0=None,
    force_multistart=False,
    use_global_fallback=False,
    maxiter_global=300,
    maxiter_local=300,
    ftol=1e-12,
    seed=1234,
    verbose=False,
):
    """
    Fast fixed-theta certified optimization.

    Variables:
        x = [rho, phi],
        rho = r/S in [0,1].

    Strategy:
        - If x0 is available and force_multistart=False:
            only use x0 and a few tiny perturbations.
        - If x0 is None or force_multistart=True:
            use deterministic multistart.
        - If use_global_fallback=True:
            run differential_evolution and polish.
    """
    coeffs = bell_coefficients_rank1(
        alpha=alpha,
        gamma=gamma,
        theta=theta,
        N=N,
    )

    if classical_slope is not None:
        energy_scale = abs(classical_slope * N)
    else:
        energy_scale = max(1.0, abs(coeffs["J"]))

    def obj_scaled(x):
        rho, phi = x
        rho = np.clip(rho, 0.0, 1.0)
        phi = wrap_phi(phi)

        return certified_f_rho_phi_fixed_theta_stable(
            rho=rho,
            phi=phi,
            coeffs=coeffs,
            N=N,
        ) / energy_scale

    bounds = [
        (0.0, 1.0),
        (-np.pi, np.pi),
    ]

    if x0 is not None and not force_multistart:
        guesses = warm_certified_guesses(x0)
    else:
        guesses = deterministic_certified_guesses(coeffs)

    best = None

    # Local optimization from selected guesses.
    for guess in guesses:
        opt = minimize(
            obj_scaled,
            x0=np.asarray(guess, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter_local,
                "ftol": ftol,
                "gtol": 1e-10,
            },
        )

        if best is None or opt.fun < best.fun:
            best = opt

    # Optional global fallback.
    if use_global_fallback:
        de = differential_evolution(
            obj_scaled,
            bounds=bounds,
            tol=1e-11,
            polish=False,
            maxiter=maxiter_global,
            updating="immediate",
            workers=1,
            seed=seed + int(N),
        )

        if best is None or de.fun < best.fun:
            best = de

        opt = minimize(
            obj_scaled,
            x0=de.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter_local,
                "ftol": ftol,
                "gtol": 1e-10,
            },
        )

        if opt.fun < best.fun:
            best = opt

    rho = float(np.clip(best.x[0], 0.0, 1.0))
    phi = float(wrap_phi(best.x[1]))

    S = N / 2.0
    r = rho * S

    Sx = r * np.sin(phi)
    Sz = r * np.cos(phi)

    beta_cert = certified_f_rho_phi_fixed_theta_stable(
        rho=rho,
        phi=phi,
        coeffs=coeffs,
        N=N,
    )

    if verbose:
        print("=" * 80)
        print(f"N = {N}")
        print("beta_cert =", beta_cert)
        print("rho =", rho)
        print("r =", r)
        print("phi =", phi)
        print("Sx =", Sx)
        print("Sz =", Sz)
        print("success =", getattr(best, "success", None))
        print("message =", getattr(best, "message", None))

    return {
        "beta_cert": beta_cert,
        "rho_cert": rho,
        "r_cert": r,
        "phi_cert": phi,
        "Sx_cert": Sx,
        "Sz_cert": Sz,
        "success_cert": getattr(best, "success", True),
        "message_cert": str(getattr(best, "message", "")),
        "x_cert": np.array([rho, phi], dtype=float),
        "coeffs": coeffs,
        "optimizer_result": best,
    }





def compute_three_ratios_rank1_dp_classical(
    alpha,
    gamma,
    N,
    theta0_params=None,
    method="BFGS",
    maxiter_quantum=1000,
    gtol=1e-8,
    maxiter_cert_global=300,
    maxiter_cert_local=300,
    ftol_cert=1e-12,
    use_abs_Jperp=True,
    use_global_fallback=True,
    verbose=True,
):
    """
    Compute three ratios for a given N, alpha, gamma:

        ratio_Q    = beta_Q / beta_C
        ratio_HP   = E_HP / beta_C
        ratio_cert = beta_cert / beta_C

    where beta_C is the exact finite-N classical bound from DP.
    """

    # ------------------------------------------------------------
    # 1. Exact finite-N classical bound by DP
    # ------------------------------------------------------------
    dp_result = classical_bounds_rank1_dp(
        alpha=alpha,
        gamma=gamma,
        Nmax=N,
        keep_argmin=True,
    )

    beta_C_fraction = dp_result["bounds"][N]
    beta_C = float(beta_C_fraction)

    # ------------------------------------------------------------
    # 2. Exact quantum value by tridiagonal optimization
    # ------------------------------------------------------------
    qres = optimize_theta_reverse_symmetry_gradient_fast(
        alpha=alpha,
        gamma=gamma,
        N=N,
        theta0_params=theta0_params,
        method=method,
        maxiter=maxiter_quantum,
        gtol=gtol,
        verbose=False,
    )

    beta_Q = qres["beta_Q"]
    theta_full = qres["theta_full"]

    # ------------------------------------------------------------
    # 3. HP variational energy at the optimized theta
    # ------------------------------------------------------------
    hp_res = hp_ground_energy_rank1(
        alpha=alpha,
        gamma=gamma,
        theta=theta_full,
        N=N,
        use_abs_Jperp=use_abs_Jperp,
    )

    E_HP = hp_res["E_HP"]

    # ------------------------------------------------------------
    # 4. Certified moment lower bound at the optimized theta
    # ------------------------------------------------------------
    cert_res = optimize_certified_fixed_theta_fast_warm(
        alpha=alpha,
        gamma=gamma,
        theta=theta_full,
        N=N,
        classical_slope=None,
        x0=None,
        force_multistart=True,
        use_global_fallback=use_global_fallback,
        maxiter_global=maxiter_cert_global,
        maxiter_local=maxiter_cert_local,
        ftol=ftol_cert,
        seed=1234,
        verbose=False,
    )

    beta_cert = cert_res["beta_cert"]

    # ------------------------------------------------------------
    # 5. Ratios
    # ------------------------------------------------------------
    ratio_Q = beta_Q / beta_C
    ratio_HP = E_HP / beta_C
    ratio_cert = beta_cert / beta_C

    result = {
        "N": N,

        # classical
        "beta_C": beta_C,
        "beta_C_fraction": beta_C_fraction,
        "classical_argmin_w": dp_result["argmins"][N],
        "compressed_u_states": len(dp_result["compressed_cost"]),
        "final_w_states": len(dp_result["last_dp"]),

        # quantum exact
        "beta_Q": beta_Q,
        "ratio_Q": ratio_Q,
        "theta_params": qres["theta_params"],
        "theta_full": theta_full,
        "grad_norm": qres["grad_norm"],
        "quantum_success": qres["success"],
        "quantum_message": qres["message"],

        # HP
        "E_HP": E_HP,
        "ratio_HP": ratio_HP,
        "E_HP_minus_beta_Q": E_HP - beta_Q,
        "ratio_HP_minus_ratio_Q": ratio_HP - ratio_Q,
        "J_perp_raw": hp_res["J_perp_raw"],
        "K_perp": hp_res["K_perp"],
        "J_gamma": hp_res["J_gamma"],
        "gamma2": hp_res["gamma2"],
        "HP_denom": hp_res["HP_denom"],

        # certified
        "beta_cert": beta_cert,
        "ratio_cert": ratio_cert,
        "beta_cert_minus_beta_Q": beta_cert - beta_Q,
        "ratio_cert_minus_ratio_Q": ratio_cert - ratio_Q,
        "rho_cert": cert_res["rho_cert"],
        "r_cert": cert_res["r_cert"],
        "phi_cert": cert_res["phi_cert"],
        "Sx_cert": cert_res["Sx_cert"],
        "Sz_cert": cert_res["Sz_cert"],
        "cert_success": cert_res["success_cert"],
        "cert_message": cert_res["message_cert"],
    }

    if verbose:
        print("=" * 100)
        print(f"N = {N}")
        print("alpha =", alpha)
        print("gamma =", gamma)
        print("-" * 100)

        print("Classical bound:")
        print("  beta_C =", beta_C_fraction, "≈", beta_C)
        print("  argmin w =", dp_result["argmins"][N])
        print()

        print("Optimized quantum value:")
        print("  beta_Q =", beta_Q)
        print("  ratio_Q = beta_Q / beta_C =", ratio_Q)
        print("  theta =", theta_full)
        print("  grad_norm =", qres["grad_norm"])
        print()

        print("HP approximation:")
        print("  E_HP =", E_HP)
        print("  ratio_HP = E_HP / beta_C =", ratio_HP)
        print("  E_HP - beta_Q =", E_HP - beta_Q)
        print("  ratio_HP - ratio_Q =", ratio_HP - ratio_Q)
        print()

        print("Certified moment bound:")
        print("  beta_cert =", beta_cert)
        print("  ratio_cert = beta_cert / beta_C =", ratio_cert)
        print("  beta_cert - beta_Q =", beta_cert - beta_Q)
        print("  ratio_cert - ratio_Q =", ratio_cert - ratio_Q)
        print()

        print("Expected energy ordering:")
        print("  beta_cert <= beta_Q <= E_HP")
        print("  values:", beta_cert, "<=", beta_Q, "<=", E_HP)

        if beta_C < 0:
            print("Since beta_C < 0, expected ratio ordering:")
            print("  ratio_cert >= ratio_Q >= ratio_HP")
            print("  values:", ratio_cert, ">=", ratio_Q, ">=", ratio_HP)
        elif beta_C > 0:
            print("Since beta_C > 0, expected ratio ordering:")
            print("  ratio_cert <= ratio_Q <= ratio_HP")
            print("  values:", ratio_cert, "<=", ratio_Q, "<=", ratio_HP)

    return result


def scan_three_ratios_rank1_dp_classical(
    alpha,
    gamma,
    N_list,
    theta0_params=None,
    method="BFGS",
    maxiter_quantum=1000,
    gtol=1e-8,
    maxiter_cert_global=300,
    maxiter_cert_local=300,
    ftol_cert=1e-12,
    use_abs_Jperp=True,
    use_global_fallback=True,
    verbose=True,
):
    results = []
    current_theta0 = theta0_params

    for N in N_list:
        res = compute_three_ratios_rank1_dp_classical(
            alpha=alpha,
            gamma=gamma,
            N=N,
            theta0_params=current_theta0,
            method=method,
            maxiter_quantum=maxiter_quantum,
            gtol=gtol,
            maxiter_cert_global=maxiter_cert_global,
            maxiter_cert_local=maxiter_cert_local,
            ftol_cert=ftol_cert,
            use_abs_Jperp=use_abs_Jperp,
            use_global_fallback=use_global_fallback,
            verbose=False,
        )

        current_theta0 = res["theta_params"]
        results.append(res)

        if verbose:
            print(
                f"N={N:4d}, "
                f"beta_C={res['beta_C']:.12g}, "
                f"beta_Q={res['beta_Q']:.12g}, "
                f"E_HP={res['E_HP']:.12g}, "
                f"beta_cert={res['beta_cert']:.12g}, "
                f"ratio_cert={res['ratio_cert']:.12f}, "
                f"ratio_Q={res['ratio_Q']:.12f}, "
                f"ratio_HP={res['ratio_HP']:.12f}, "
                f"theta={res['theta_full']}, "
                f"grad={res['grad_norm']:.2e}"
            )

    return results




# ============================================================
# Main three-ratio generator without reading CSV
# ============================================================

def compute_three_ratios_from_optimization_fast_warm(
    alpha,
    gamma,
    N_list,
    classical_slope,
    theta0_params=None,
    output_csv=None,
    method="BFGS",
    maxiter_quantum=1000,
    gtol=1e-8,
    use_abs_Jperp=True,
    maxiter_global=300,
    maxiter_local=300,
    ftol=1e-12,
    use_global_fallback=True,
    verbose=True,
):
    """
    Optimize theta for each N, then compute three ratios:

        ratio_Q    = beta_Q / beta_C
        ratio_HP   = E_HP / beta_C
        ratio_cert = beta_cert / beta_C

    where

        beta_C = classical_slope * N.

    No classical DP is computed.
    No input CSV is read.

    The quantum theta optimizer is warm-started by theta0_params from
    the previous N.

    The certified optimizer is run independently for each N.
    """

    if classical_slope is None:
        raise ValueError(
            "classical_slope must be provided, since beta_C = classical_slope * N."
        )

    rows_out = []
    current_theta0 = theta0_params

    for idx, N in enumerate(N_list):
        N = int(N)

        # ----------------------------------------------------
        # Classical bound from asymptotic slope
        # ----------------------------------------------------
        beta_C = classical_slope * N

        # ----------------------------------------------------
        # Exact quantum optimization
        # ----------------------------------------------------
        qres = optimize_theta_reverse_symmetry_gradient_fast(
            alpha=alpha,
            gamma=gamma,
            N=N,
            theta0_params=current_theta0,
            method=method,
            maxiter=maxiter_quantum,
            gtol=gtol,
            verbose=False,
        )

        beta_Q = qres["beta_Q"]
        theta_full = qres["theta_full"]
        theta_params = qres["theta_params"]

        current_theta0 = theta_params

        ratio_Q = beta_Q / beta_C

        # ----------------------------------------------------
        # HP energy at optimized theta
        # ----------------------------------------------------
        hp_res = hp_ground_energy_rank1(
            alpha=alpha,
            gamma=gamma,
            theta=theta_full,
            N=N,
            use_abs_Jperp=use_abs_Jperp,
        )

        E_HP = hp_res["E_HP"]
        ratio_HP = E_HP / beta_C

        E_HP_minus_beta_Q = E_HP - beta_Q
        ratio_HP_minus_ratio_Q = ratio_HP - ratio_Q

        # ----------------------------------------------------
        # Certified moment bound at optimized theta
        # No warm start
        # ----------------------------------------------------
        cert_res = optimize_certified_fixed_theta_fast_warm(
            alpha=alpha,
            gamma=gamma,
            theta=theta_full,
            N=N,
            classical_slope=classical_slope,
            x0=None,
            force_multistart=True,
            use_global_fallback=use_global_fallback,
            maxiter_global=maxiter_global,
            maxiter_local=maxiter_local,
            ftol=ftol,
            seed=1234,
            verbose=False,
        )

        beta_cert = cert_res["beta_cert"]
        ratio_cert = beta_cert / beta_C

        beta_cert_minus_beta_Q = beta_cert - beta_Q
        ratio_cert_minus_ratio_Q = ratio_cert - ratio_Q

        # ----------------------------------------------------
        # Ordering diagnostics
        # ----------------------------------------------------
        energy_order_ok = (
            beta_cert <= beta_Q + 1e-7
            and beta_Q <= E_HP + 1e-7
        )

        if beta_C < 0:
            ratio_order_ok = (
                ratio_cert >= ratio_Q - 1e-10
                and ratio_Q >= ratio_HP - 1e-10
            )
        elif beta_C > 0:
            ratio_order_ok = (
                ratio_cert <= ratio_Q + 1e-10
                and ratio_Q <= ratio_HP + 1e-10
            )
        else:
            ratio_order_ok = False

        # ----------------------------------------------------
        # Put the three ratios at the beginning of each row
        # ----------------------------------------------------
        row = {
            # Main output columns
            "N": N,
            "ratio_cert": ratio_cert,
            "ratio_Q": ratio_Q,
            "ratio_HP": ratio_HP,

            # Alias, useful for compatibility with previous code
            "ratio": ratio_Q,

            # Classical slope bound
            "classical_slope": classical_slope,
            "beta_C": beta_C,

            # Energies
            "beta_cert": beta_cert,
            "beta_Q": beta_Q,
            "E_HP": E_HP,

            # Differences
            "beta_cert_minus_beta_Q": beta_cert_minus_beta_Q,
            "E_HP_minus_beta_Q": E_HP_minus_beta_Q,
            "ratio_cert_minus_ratio_Q": ratio_cert_minus_ratio_Q,
            "ratio_HP_minus_ratio_Q": ratio_HP_minus_ratio_Q,

            # Quantum optimization
            "theta_params": theta_params,
            "theta_full": theta_full,
            "grad_norm": qres["grad_norm"],
            "quantum_success": qres["success"],
            "quantum_message": qres["message"],

            # HP details
            "J_perp_raw": hp_res["J_perp_raw"],
            "K_perp": hp_res["K_perp"],
            "J_gamma": hp_res["J_gamma"],
            "gamma2": hp_res["gamma2"],
            "HP_denom": hp_res["HP_denom"],

            # Certified details
            "rho_cert": cert_res["rho_cert"],
            "r_cert": cert_res["r_cert"],
            "phi_cert": cert_res["phi_cert"],
            "Sx_cert": cert_res["Sx_cert"],
            "Sz_cert": cert_res["Sz_cert"],
            "success_cert": cert_res["success_cert"],
            "message_cert": cert_res["message_cert"],

            # Diagnostics
            "energy_order_ok": energy_order_ok,
            "ratio_order_ok": ratio_order_ok,
        }

        rows_out.append(row)

        if verbose:
            print(
                f"N={N:6d}, "
                f"ratio_cert={ratio_cert:.12f}, "
                f"ratio_Q={ratio_Q:.12f}, "
                f"ratio_HP={ratio_HP:.12f}, "
                f"beta_cert={beta_cert:.12g}, "
                f"beta_Q={beta_Q:.12g}, "
                f"E_HP={E_HP:.12g}, "
                f"beta_C={beta_C:.12g}, "
                f"cert-Q={ratio_cert_minus_ratio_Q:.3e}, "
                f"HP-Q={ratio_HP_minus_ratio_Q:.3e}, "
                f"rho={cert_res['rho_cert']:.8f}, "
                f"phi={cert_res['phi_cert']:.8f}, "
                f"grad={qres['grad_norm']:.2e}, "
                f"order={energy_order_ok and ratio_order_ok}"
            )

    # ----------------------------------------------------
    # Optional CSV output
    # ----------------------------------------------------
    if output_csv is not None:
        fieldnames_out = [
            # Put the three ratios at the front
            "N",
            "ratio_cert",
            "ratio_Q",
            "ratio_HP",
            "ratio",

            # Classical slope bound
            "classical_slope",
            "beta_C",

            # Energies
            "beta_cert",
            "beta_Q",
            "E_HP",

            # Differences
            "beta_cert_minus_beta_Q",
            "E_HP_minus_beta_Q",
            "ratio_cert_minus_ratio_Q",
            "ratio_HP_minus_ratio_Q",

            # Quantum optimization
            "theta_params",
            "theta_full",
            "grad_norm",
            "quantum_success",
            "quantum_message",

            # HP details
            "J_perp_raw",
            "K_perp",
            "J_gamma",
            "gamma2",
            "HP_denom",

            # Certified details
            "rho_cert",
            "r_cert",
            "phi_cert",
            "Sx_cert",
            "Sz_cert",
            "success_cert",
            "message_cert",

            # Diagnostics
            "energy_order_ok",
            "ratio_order_ok",
        ]

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_out)
            writer.writeheader()

            for row in rows_out:
                row_out = {}

                for key in fieldnames_out:
                    value = row.get(key, "")

                    if isinstance(value, np.ndarray):
                        value = value.tolist()

                    row_out[key] = value

                writer.writerow(row_out)

        if verbose:
            print("=" * 100)
            print(f"Wrote {output_csv}")
            print(f"Rows = {len(rows_out)}")

    return rows_out

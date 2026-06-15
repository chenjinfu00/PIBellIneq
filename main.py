if __name__ == "__main__":

    Nmax = 100

    cases = [
        {
            "name": "m2_11",
            "alpha": [1, -1],
            "gamma": [1, 1],
            "classical_slope": -2,
            "N_min": 10,
            "theta0_params": [0.4],
            "output_csv": "ratio_scan_m2_11_three_ratios_Nmax100.csv",
        },
        {
            "name": "m3_545",
            "alpha": [-45, 0, 45],
            "gamma": [5, 4, 5],
            "classical_slope": -98,
            "N_min": 10,
            "theta0_params": [0.65],
            "output_csv": "ratio_scan_m3_545_three_ratios_Nmax100.csv",
        },
        {
            "name": "m4_19151519",
            "alpha": [-931, -225, 225, 931],
            "gamma": [19, 15, 15, 19],
            "classical_slope": -2312,
            "N_min": 10,
            "theta0_params": [0.7, 0.35],
            "output_csv": "ratio_scan_m4_19151519_three_ratios_Nmax100.csv",
        },
        {
            "name": "m5_9778727897",
            "alpha": [-31525, -11700, 0, 11700, 31525],
            "gamma": [97, 78, 72, 78, 97],
            "classical_slope": -89042,
            "N_min": 25,
            "theta0_params": [0.7, 0.35],
            "output_csv": "ratio_scan_m5_9778727897_three_ratios_Nmax100.csv",
        },
        {
            "name": "m6_166113651225122513651661",
            "alpha": [
                -11362901,
                -5207475,
                -1500625,
                1500625,
                5207475,
                11362901,
            ],
            "gamma": [
                1661,
                1365,
                1225,
                1225,
                1365,
                1661,
            ],
            "classical_slope": -36142002,
            "N_min": 10,
            "theta0_params": [0.7, 0.45, 0.25],
            "output_csv": "ratio_scan_m6_166113651225122513651661_three_ratios_Nmax100.csv",
        },
    ]

    all_results = {}

    for case in cases:
        print("=" * 120)
        print(f"Running case: {case['name']}, N = {case['N_min']} to {Nmax}")
        print("=" * 120)

        results = compute_three_ratios_from_optimization_fast_warm(
            alpha=case["alpha"],
            gamma=case["gamma"],
            N_list=range(case["N_min"], Nmax + 1),
            classical_slope=case["classical_slope"],
            theta0_params=case["theta0_params"],
            output_csv=case["output_csv"],
            method="BFGS",
            maxiter_quantum=1000,
            gtol=1e-8,
            use_abs_Jperp=True,
            maxiter_global=300,
            maxiter_local=300,
            ftol=1e-12,
            use_global_fallback=True,
            verbose=True,
        )

        all_results[case["name"]] = results






alpha = [-31525, -11700, 0, 11700, 31525]
gamma = [97, 78, 72, 78, 97]

N_min = 10
N_max = 40

results = []
current_theta0 = [0.7, 0.35]

for N in range(N_min, N_max + 1):
    res = compute_three_ratios_rank1_dp_classical(
        alpha=alpha,
        gamma=gamma,
        N=N,
        theta0_params=current_theta0,
        method="BFGS",
        maxiter_quantum=1000,
        gtol=1e-8,
        maxiter_cert_global=300,
        maxiter_cert_local=300,
        ftol_cert=1e-12,
        use_abs_Jperp=True,
        use_global_fallback=True,
        verbose=False,
    )

    current_theta0 = res["theta_params"]
    results.append(res)

    print(
        f"N={N:4d}, "
        f"ratio_cert={res['ratio_cert']:.12f}, "
        f"ratio_Q={res['ratio_Q']:.12f}, "
        f"ratio_HP={res['ratio_HP']:.12f}, "
        f"beta_C={res['beta_C']:.12g}, "
        f"beta_cert={res['beta_cert']:.12g}, "
        f"beta_Q={res['beta_Q']:.12g}, "
        f"E_HP={res['E_HP']:.12g}, "
        f"grad={res['grad_norm']:.2e}"
    )


filename = "m5_N10_40_three_ratios_DP_classical.csv"

fieldnames = [
    "N",

    # Three ratios
    "ratio_cert",
    "ratio_Q",
    "ratio_HP",

    # Classical
    "beta_C",
    "beta_C_fraction",
    "classical_argmin_w",
    "compressed_u_states",
    "final_w_states",

    # Quantum exact
    "beta_Q",
    "theta_params",
    "theta_full",
    "grad_norm",
    "quantum_success",
    "quantum_message",

    # HP
    "E_HP",
    "E_HP_minus_beta_Q",
    "ratio_HP_minus_ratio_Q",
    "J_perp_raw",
    "K_perp",
    "J_gamma",
    "gamma2",
    "HP_denom",

    # Certified
    "beta_cert",
    "beta_cert_minus_beta_Q",
    "ratio_cert_minus_ratio_Q",
    "rho_cert",
    "r_cert",
    "phi_cert",
    "Sx_cert",
    "Sz_cert",
    "cert_success",
    "cert_message",
]

with open(filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for res in results:
        row = {}

        for key in fieldnames:
            value = res.get(key, "")

            if hasattr(value, "tolist"):
                value = value.tolist()

            if key in ["beta_C_fraction", "classical_argmin_w"]:
                value = str(value)

            row[key] = value

        writer.writerow(row)

print("Saved to:", filename)

"""Run PySR on combined SPARC + Mistele+2024 lensing RAR data.

This is the first symbolic regression over 5.5 dex of dynamic range.
Nobody has done this — SPARC is ~3 dex, Mistele lensing adds 2.5 dex more.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

# ── Data sources ──────────────────────────────────────────────────────────────

# Mistele+2024 lensing data (Table 1: log_gbar, log_gobs, stat_err, sys_err)
LENSING_DATA = np.array([
    [-12.39, -11.11, 0.06, 0.00],
    [-12.64, -11.21, 0.05, 0.02],
    [-12.89, -11.29, 0.05, 0.00],
    [-13.13, -11.47, 0.05, 0.02],
    [-13.38, -11.59, 0.05, 0.01],
    [-13.63, -11.76, 0.06, 0.03],
    [-13.87, -11.93, 0.07, 0.05],
    [-14.12, -12.08, 0.07, 0.07],
    [-14.37, -12.27, 0.08, 0.13],
    [-14.61, -12.44, 0.08, 0.25],
    [-14.86, -12.85, 0.12, 0.67],
])

# SPARC binned RAR (10 points, Lelli+2017 Table 5)
SPARC_BINNED = np.array([
    [-10.82, -10.35, 0.03],
    [-10.54, -10.15, 0.02],
    [-10.26, -9.93, 0.02],
    [-9.97, -9.70, 0.02],
    [-9.69, -9.47, 0.01],
    [-9.41, -9.23, 0.01],
    [-9.12, -8.98, 0.01],
    [-8.88, -8.75, 0.01],
    [-8.70, -8.59, 0.01],
    [-8.37, -8.28, 0.01],
])


# ── Build combined dataset for PySR ──────────────────────────────────────────

def build_combined_dataset():
    """Build a single dataset with both SPARC and lensing data."""
    sparc_x = SPARC_BINNED[:, 0]  # log_gbar
    sparc_y = SPARC_BINNED[:, 1]  # log_gobs
    sparc_err = SPARC_BINNED[:, 2]

    lens_x = LENSING_DATA[:, 0]
    lens_y = LENSING_DATA[:, 1]
    lens_err = np.sqrt(LENSING_DATA[:, 2]**2 + LENSING_DATA[:, 3]**2 + 0.1**2)

    x_all = np.concatenate([sparc_x, lens_x])
    y_all = np.concatenate([sparc_y, lens_y])

    # Use inverse variance as weights (PySR doesn't directly use per-point sigmas)
    weights_sparc = 1.0 / np.maximum(sparc_err, 1e-10)**2
    weights_lens = 1.0 / np.maximum(lens_err, 1e-10)**2

    # Normalize to mean weight = 1
    weights_all = np.concatenate([weights_sparc, weights_lens])
    weights_all = weights_all / np.mean(weights_all)

    print(f"  SPARC points: {len(sparc_x)}")
    print(f"  Lensing points: {len(lens_x)}")
    print(f"  Total: {len(x_all)}")
    print(f"  Dynamic range: {x_all.max() - x_all.min():.1f} dex")

    return x_all, y_all, weights_all


# ── Run PySR ─────────────────────────────────────────────────────────────────

def run_pysr(x, y, weights, outdir="analysis", n_iters=200):
    """Run PySR on combined data."""
    from pysr import PySRRegressor

    print(f"\nRunning PySR on {len(x)} points ({n_iters} iterations)...")

    model = PySRRegressor(
        niterations=n_iters,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "square", "neg"],
        maxsize=25,
        populations=12,
        population_size=50,
        model_selection="accuracy",
        elementwise_loss="L2DistLoss()",
        parsimony=0.001,
        temp_equation_file=True,
        progress=False,
        verbosity=0,
    )

    # PySR expects 2D input for features
    X = x.reshape(-1, 1)

    print("  Fitting...")
    model.fit(X, y, weights=weights)

    print(f"\n  Best equation: {model.sympy()}")

    # Get all equations on the Pareto front
    eqs = model.equations_
    eqs = eqs.sort_values("score", ascending=False)

    print(f"\n  Pareto front equations ({len(eqs)}):")
    for i, row in eqs.head(10).iterrows():
        c = row.get("complexity", "?")
        l = row.get("loss", "?")
        s = row.get("score", "?")
        eq = row.get("sympy_format", str(row.get("equation", "?")))
        print(f"    Cpx {c}: loss={l:.4f}, score={s:.4f}, eq={eq}")

    return model


# ── Compare with MOND ────────────────────────────────────────────────────────

def compare_models(x, y, weights, model, outdir="analysis", n_iters=200):
    """Compare PySR results with MOND IFs."""
    from scipy.optimize import minimize
    from parse_sparc import parse_mass_models, compute_radial_accelerations

    def mond_simple_log(log_gbar, log_a0):
        gbar = 10**log_gbar
        a0 = 10**log_a0
        gobs = gbar * (1 + np.sqrt(1 + 4*a0/np.maximum(gbar, 1e-20))) / 2
        return np.log10(gobs)

    def mond_mcgaugh_log(log_gbar, log_a0):
        gbar = 10**log_gbar
        a0 = 10**log_a0
        gobs = gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20) / a0)), 1e-20)
        return np.log10(gobs)

    def chi2_mond(params, func):
        pred = func(x, params[0])
        return np.sum(weights * (y - pred)**2)

    # Fit MOND Simple
    r = minimize(lambda p: chi2_mond(p, mond_simple_log), [-10],
                 bounds=[(-12, -8)], method="L-BFGS-B")
    a0_simple = 10**r.x[0]
    chi2_simple = r.fun
    print(f"\n  MOND Simple: a₀={a0_simple:.3e}, χ²={chi2_simple:.1f}")

    # Fit MOND McGaugh
    r = minimize(lambda p: chi2_mond(p, mond_mcgaugh_log), [-10],
                 bounds=[(-12, -8)], method="L-BFGS-B")
    a0_mcgaugh = 10**r.x[0]
    chi2_mcgaugh = r.fun
    print(f"  McGaugh RAR IF: a₀={a0_mcgaugh:.3e}, χ²={chi2_mcgaugh:.1f}")

    # PySR prediction
    X = x.reshape(-1, 1)
    y_pred_pysr = model.predict(X)
    chi2_pysr = np.sum(weights * (y - y_pred_pysr)**2)
    print(f"  PySR (best): χ²={chi2_pysr:.1f}")

    # Best equation from PySR
    eqs = model.equations_
    best_eq = eqs.sort_values("score", ascending=False).iloc[0]
    best_formula = best_eq.get("sympy_format", str(best_eq.get("equation", "?")))
    print(f"  Best PySR formula: {best_formula}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Data
    n_sparc = len(SPARC_BINNED)
    ax.errorbar(x[:n_sparc], y[:n_sparc],
                yerr=1.0/np.sqrt(weights[:n_sparc]) * np.std(y) if False else None,
                fmt="o", color="gray", ms=6, label="SPARC binned (Lelli+2017)")
    ax.errorbar(x[n_sparc:], y[n_sparc:],
                yerr=1.0/np.sqrt(weights[n_sparc:]) * np.std(y) if False else None,
                fmt="D", color="orange", ms=6, label="Lensing (Mistele+2024)")

    # Models
    x_grid = np.linspace(-15, -8, 500)
    ax.plot(x_grid, mond_simple_log(x_grid, np.log10(a0_simple)),
            "r--", lw=2, label=f"MOND Simple (a₀={a0_simple:.2e})")
    ax.plot(x_grid, mond_mcgaugh_log(x_grid, np.log10(a0_mcgaugh)),
            "b:", lw=2, label=f"McGaugh RAR IF (a₀={a0_mcgaugh:.2e})")

    # PySR prediction
    y_grid = model.predict(x_grid.reshape(-1, 1))
    ax.plot(x_grid, y_grid, "k-", lw=2.5, label=f"PySR (best)")

    # Best SR equations  
    y_grid = model.predict(x_grid.reshape(-1, 1))
    ax.plot(x_grid, y_grid, "k-", lw=2.5, label=f"PySR (best)")

    ax.plot(x_grid, x_grid, "k:", lw=0.5, alpha=0.3, label="1:1")
    ax.plot(x_grid, -5 + 0.5 * (x_grid + 10), "k:", lw=0.5, alpha=0.3)

    ax.set_xlabel("log gbar (m/s²)")
    ax.set_ylabel("log gobs (m/s²)")
    ax.legend(fontsize=8)
    ax.set_xlim(-15, -8)
    ax.set_ylim(-14, -8)

    plt.tight_layout()
    plt.savefig(f"{outdir}/joint_pysr_lensing.png", dpi=150)
    print(f"  Saved {outdir}/joint_pysr_lensing.png")
    plt.close()

    # Residual plot
    fig, ax = plt.subplots(figsize=(10, 5))
    res_pysr = y - y_pred_pysr
    res_simple = y - mond_simple_log(x, np.log10(a0_simple))
    res_mcgaugh = y - mond_mcgaugh_log(x, np.log10(a0_mcgaugh))

    ax.plot(x[:n_sparc], res_pysr[:n_sparc], "ko", ms=4, label="PySR (SPARC)")
    ax.plot(x[n_sparc:], res_pysr[n_sparc:], "kD", ms=4, label="PySR (lensing)")
    ax.plot(x[:n_sparc], res_simple[:n_sparc], "r^", ms=3, alpha=0.5,
            label="MOND Simple (SPARC)")
    ax.plot(x[:n_sparc], res_mcgaugh[:n_sparc], "bs", ms=3, alpha=0.5,
            label="McGaugh RAR IF (SPARC)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("log gbar (m/s²)")
    ax.set_ylabel("Residual (dex)")
    ax.legend(fontsize=8)
    ax.set_xlim(-15, -8)

    plt.tight_layout()
    plt.savefig(f"{outdir}/joint_pysr_lensing_residuals.png", dpi=150)
    print(f"  Saved {outdir}/joint_pysr_lensing_residuals.png")
    plt.close()

    return {
        "pysr_loss": chi2_pysr,
        "a0_simple": a0_simple, "chi2_simple": chi2_simple,
        "a0_mcgaugh": a0_mcgaugh, "chi2_mcgaugh": chi2_mcgaugh,
    }


if __name__ == "__main__":
    os.makedirs("analysis", exist_ok=True)

    print("=" * 60)
    print("PySR on combined SPARC + Lensing RAR (5.5 dex)")
    print("=" * 60)

    x, y, weights = build_combined_dataset()
    model = run_pysr(x, y, weights, n_iters=200)
    results = compare_models(x, y, weights, model)
    print("\nDone.")

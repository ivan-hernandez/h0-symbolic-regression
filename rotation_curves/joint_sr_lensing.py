"""Joint fit of SR forms to SPARC + Mistele+2024 weak-lensing RAR data.

Mistele+2024 lensing extends the RAR by 2.5 dex to gbar ~ 10^{-14} m/s^2.
Our CPX5 form fails at low g (asymptotic constant vs √gbar).
We test if any SR form can describe the full 5.5 dex range.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from parse_sparc import parse_mass_models, compute_radial_accelerations

# ── Mistele+2024 lensing data (Table 1, log10 gbar, log10 gobs, stat_err, sys_err) ──
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

# SPARC binned RAR data (Lelli+2017, from RARbins.mrt)
# log_gbar, log_gobs, error
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


# ── Models ────────────────────────────────────────────────────────────────────

def mond_mcgaugh(gbar, a0):
    """McGaugh RAR IF: gobs = gbar / (1 - exp(-sqrt(gbar/a0)))"""
    return gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20) / a0)), 1e-20)

def mond_mcgaugh_log(log_gbar, log_a0):
    return np.log10(mond_mcgaugh(10**log_gbar, 10**log_a0))

def cpx5_log(x, a, b):
    return a + b / x

def simple_power_law_log(x, alpha, log_c):
    """gobs = c * gbar^alpha  →  log_gobs = log_c + alpha * log_gbar"""
    return log_c + alpha * x

def broken_power_law_log(x, alpha_low, alpha_high, log_gbar_break, log_c):
    """Two-segment power law in log-log space."""
    x_break = log_gbar_break
    y_at_break = log_c + alpha_low * x_break
    return np.where(x < x_break,
                    log_c + alpha_low * x,
                    y_at_break + alpha_high * (x - x_break))


# ── Joint fit ─────────────────────────────────────────────────────────────────

def joint_fit(outdir="analysis"):
    """Fit models jointly to SPARC kinematic + lensing data."""
    print("=" * 60)
    print("Joint SR + Lensing Fit")
    print("=" * 60)

    # Combine data
    sparc_x = SPARC_BINNED[:, 0]
    sparc_y = SPARC_BINNED[:, 1]
    sparc_err = SPARC_BINNED[:, 2]

    lens_x = LENSING_DATA[:, 0]
    lens_y = LENSING_DATA[:, 1]
    lens_err = np.sqrt(LENSING_DATA[:, 2]**2 + LENSING_DATA[:, 3]**2 + 0.1**2)
    # Add 0.1 dex systematic from stellar mass uncertainties (per Mistele+2024)

    x_all = np.concatenate([sparc_x, lens_x])
    y_all = np.concatenate([sparc_y, lens_y])
    err_all = np.concatenate([sparc_err, lens_err])

    print(f"  SPARC binned points: {len(sparc_x)}")
    print(f"  Lensing points: {len(lens_x)}")
    print(f"  Dynamic range: {x_all.max() - x_all.min():.1f} dex")

    # 1. McGaugh RAR IF
    def chi2_mond(params):
        a0 = 10**params[0]
        pred_log = np.log10(mond_mcgaugh(10**x_all, a0))
        return np.sum(((y_all - pred_log) / err_all)**2)

    result_mond = minimize(chi2_mond, x0=[-10], bounds=[(-12, -8)], method="L-BFGS-B")
    log_a0_mond = result_mond.x[0]
    chi2_mond_val = result_mond.fun
    n_params_mond = 1
    aic_mond = chi2_mond_val + 2 * n_params_mond
    print(f"\n  McGaugh RAR IF:")
    print(f"    a₀ = {10**log_a0_mond:.3e} m/s²")
    print(f"    χ² = {chi2_mond_val:.1f} (dof = {len(x_all) - n_params_mond})")
    print(f"    AIC = {aic_mond:.1f}")

    # 2. CPX5 (our SR form)
    def chi2_cpx5(params):
        a, b = params
        pred = cpx5_log(x_all, a, b)
        return np.sum(((y_all - pred) / err_all)**2)

    result_cpx5 = minimize(chi2_cpx5, x0=[-12, -50], bounds=[(-30, 10), (-200, 200)],
                            method="L-BFGS-B")
    a_cpx5, b_cpx5 = result_cpx5.x
    chi2_cpx5_val = result_cpx5.fun
    aic_cpx5 = chi2_cpx5_val + 2 * 2
    print(f"\n  CPX5 (SR form):")
    print(f"    a = {a_cpx5:.3f}, b = {b_cpx5:.3f}")
    print(f"    χ² = {chi2_cpx5_val:.1f} (dof = {len(x_all) - 2})")
    print(f"    AIC = {aic_cpx5:.1f}")

    # 3. Simple power law
    def chi2_pl(params):
        alpha, log_c = params
        pred = simple_power_law_log(x_all, alpha, log_c)
        return np.sum(((y_all - pred) / err_all)**2)

    result_pl = minimize(chi2_pl, x0=[0.5, -5], bounds=[(0.1, 1.5), (-10, -1)],
                          method="L-BFGS-B")
    alpha_pl, log_c_pl = result_pl.x
    chi2_pl_val = result_pl.fun
    aic_pl = chi2_pl_val + 2 * 2
    print(f"\n  Simple power law:")
    print(f"    α = {alpha_pl:.3f}, c = {10**log_c_pl:.3e}")
    print(f"    χ² = {chi2_pl_val:.1f} (dof = {len(x_all) - 2})")
    print(f"    AIC = {aic_pl:.1f}")

    # 4. Broken power law
    def chi2_broken(params):
        alpha_low, alpha_high, log_gbar_break, log_c = params
        pred = broken_power_law_log(x_all, alpha_low, alpha_high, log_gbar_break, log_c)
        return np.sum(((y_all - pred) / err_all)**2)

    result_broken = minimize(chi2_broken, x0=[0.5, 1.0, -10.5, -5],
                              bounds=[(0.1, 1.5), (0.5, 1.5), (-12, -9), (-10, -1)],
                              method="L-BFGS-B")
    alpha_low_b, alpha_high_b, log_break_b, log_c_b = result_broken.x
    chi2_broken_val = result_broken.fun
    aic_broken = chi2_broken_val + 2 * 4
    print(f"\n  Broken power law:")
    print(f"    α_low = {alpha_low_b:.3f}, α_high = {alpha_high_b:.3f}")
    print(f"    break at log_gbar = {log_break_b:.2f} ({10**log_break_b:.2e} m/s²)")
    print(f"    χ² = {chi2_broken_val:.1f} (dof = {len(x_all) - 4})")
    print(f"    AIC = {aic_broken:.1f}")

    # Summary
    print(f"\n{'='*60}")
    print("Model Comparison (joint SPARC + lensing)")
    print(f"{'='*60}")
    models = [
        ("McGaugh RAR IF (1 param)", chi2_mond_val, 1, aic_mond),
        ("CPX5 (2 params)", chi2_cpx5_val, 2, aic_cpx5),
        ("Power law (2 params)", chi2_pl_val, 2, aic_pl),
        ("Broken power law (4 params)", chi2_broken_val, 4, aic_broken),
    ]
    print(f"  {'Model':<30s} {'χ²':<10s} {'k':<5s} {'AIC':<10s} {'ΔAIC':<10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*5} {'-'*10} {'-'*10}")
    best_aic = min(m[3] for m in models)
    for name, chi2, k, aic in models:
        print(f"  {name:<30s} {chi2:<10.1f} {k:<5d} {aic:<10.1f} {aic - best_aic:<+10.1f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    # Data
    ax.errorbar(sparc_x, sparc_y, yerr=sparc_err, fmt="o", color="gray",
                ms=5, label="SPARC (Lelli+2017)", capsize=2)
    ax.errorbar(lens_x, lens_y, yerr=lens_err, fmt="D", color="orange",
                ms=5, label="Lensing (Mistele+2024)", capsize=2)

    # Models
    x_grid = np.linspace(-15, -8, 300)
    ax.plot(x_grid, np.log10(mond_mcgaugh(10**x_grid, 10**log_a0_mond)),
            "r-", lw=2, label="McGaugh RAR IF")
    ax.plot(x_grid, cpx5_log(x_grid, a_cpx5, b_cpx5),
            "b--", lw=2, label="CPX5 (this work)")
    ax.plot(x_grid, simple_power_law_log(x_grid, alpha_pl, log_c_pl),
            "g-.", lw=2, label=f"Power law α={alpha_pl:.2f}")
    pred_broken = broken_power_law_log(x_grid, alpha_low_b, alpha_high_b, log_break_b, log_c_b)
    ax.plot(x_grid, pred_broken, "m:", lw=2, label="Broken PL")

    # MOND √gbar line
    ax.plot(x_grid, -5 + 0.5 * (x_grid + 10), "k:", lw=0.5, alpha=0.5, label="√g (slope 1/2)")

    ax.set_xlabel("log gbar (m/s²)")
    ax.set_ylabel("log gobs (m/s²)")
    ax.legend(fontsize=8)
    ax.set_xlim(-15, -8)
    ax.set_ylim(-14, -8)

    # Residuals
    ax = axes[1]
    for label, x, y, err, color, marker in [
        ("SPARC", sparc_x, sparc_y, sparc_err, "gray", "o"),
        ("Lensing", lens_x, lens_y, lens_err, "orange", "D"),
    ]:
        res_mond = y - np.log10(mond_mcgaugh(10**x, 10**log_a0_mond))
        res_cpx5 = y - cpx5_log(x, a_cpx5, b_cpx5)
        ax.plot(x, res_mond, marker + "-", color=color, ms=4, lw=0, alpha=0.7,
                label=f"{label} - MOND" if label == "SPARC" else None)
        ax.plot(x, res_cpx5, marker + "--", color=color, ms=4, lw=0, alpha=0.5,
                label=f"{label} - CPX5" if label == "SPARC" else None)

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("log gbar (m/s²)")
    ax.set_ylabel("Residual (dex)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-15, -8)

    plt.tight_layout()
    plt.savefig(f"{outdir}/joint_sr_lensing.png", dpi=150)
    print(f"\n  Saved {outdir}/joint_sr_lensing.png")
    plt.close()

    return {
        "mond": {"log_a0": log_a0_mond, "chi2": chi2_mond_val, "aic": aic_mond},
        "cpx5": {"a": a_cpx5, "b": b_cpx5, "chi2": chi2_cpx5_val, "aic": aic_cpx5},
        "power_law": {"alpha": alpha_pl, "log_c": log_c_pl, "chi2": chi2_pl_val, "aic": aic_pl},
        "broken_pl": {"alpha_low": alpha_low_b, "alpha_high": alpha_high_b,
                       "break": 10**log_break_b, "chi2": chi2_broken_val, "aic": aic_broken},
    }


if __name__ == "__main__":
    import os
    os.makedirs("analysis", exist_ok=True)
    results = joint_fit()
    print("\nDone.")

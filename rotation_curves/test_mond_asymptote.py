"""Test if CPX5 + MOND asymptote improves the fit.

Fits log_gobs = a + b/log_gbar + c*log_gbar and tests if c = 0 (CPX5 wins)
or c ≈ 0.5 (MOND √gbar asymptote wins), using:
1. Full SPARC mass model data (3391 pts)
2. Combined SPARC + Mistele lensing data
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from parse_sparc import parse_mass_models, compute_radial_accelerations

# ── Lensing data (Mistele+2024, Table 1) ──────────────────────────────────────
LENSING = np.array([
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

# SPARC binned data
SPARC_BINNED = np.array([
    [-10.82, -10.35, 0.03], [-10.54, -10.15, 0.02],
    [-10.26, -9.93, 0.02], [-9.97, -9.70, 0.02],
    [-9.69, -9.47, 0.01], [-9.41, -9.23, 0.01],
    [-9.12, -8.98, 0.01], [-8.88, -8.75, 0.01],
    [-8.70, -8.59, 0.01], [-8.37, -8.28, 0.01],
])

# ── Models ────────────────────────────────────────────────────────────────────

def cpx5_log(x, a, b):
    return a + b / x

def cpx5_mond_log(x, a, b, c):
    """CPX5 + MOND term: at low gbar, c*log_gbar dominates → slope = c"""
    return a + b / x + c * x

def mond_mcgaugh_log(log_gbar, log_a0):
    gbar = 10**log_gbar
    a0 = 10**log_a0
    gobs = gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20) / a0)), 1e-20)
    return np.log10(gobs)


def run_test(name, x, y, weights):
    """Fit all models to a given dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name} ({len(x)} points, {x.max()-x.min():.1f} dex)")
    print(f"{'='*60}")

    def chi2(params, func):
        return np.sum(weights * (y - func(x, *params))**2)

    def aic(chi2, k):
        return chi2 + 2 * k

    # 1. CPX5 (2 params)
    r = minimize(lambda p: chi2(p, cpx5_log), [-17, -75], method="L-BFGS-B")
    a_cpx5, b_cpx5 = r.x
    chi2_cpx5 = r.fun
    aic_cpx5 = aic(chi2_cpx5, 2)
    print(f"\n  CPX5 (2 params):")
    print(f"    log_gobs = {a_cpx5:.4f} + ({b_cpx5:.4f}) / log_gbar")
    print(f"    χ² = {chi2_cpx5:.2f}, AIC = {aic_cpx5:.1f}")

    # 2. CPX5 + MOND term (3 params) — test if c deviates from 0
    r = minimize(lambda p: chi2(p, cpx5_mond_log), [-17, -75, 0.1], method="L-BFGS-B")
    a_c5m, b_c5m, c_c5m = r.x
    chi2_c5m = r.fun
    aic_c5m = aic(chi2_c5m, 3)
    print(f"\n  CPX5 + MOND term (3 params):")
    print(f"    log_gobs = {a_c5m:.4f} + ({b_c5m:.4f}) / log_gbar + ({c_c5m:.4f})·log_gbar")
    print(f"    χ² = {chi2_c5m:.2f}, AIC = {aic_c5m:.1f}")
    print(f"    Asymptotic low-g slope c = {c_c5m:.4f}")
    print(f"    MOND predicts c = 0.5, CPX5 predicts c = 0")
    print(f"    ΔAIC vs CPX5: {aic_c5m - aic_cpx5:.1f}")

    # 3. MOND RAR IF (1 param)
    r = minimize(lambda p: chi2(p, mond_mcgaugh_log), [-10],
                 bounds=[(-12, -8)], method="L-BFGS-B")
    log_a0 = r.x[0]
    a0 = 10**log_a0
    chi2_mond = r.fun
    aic_mond = aic(chi2_mond, 1)
    print(f"\n  McGaugh RAR IF (1 param):")
    print(f"    a₀ = {a0:.3e}")
    print(f"    χ² = {chi2_mond:.2f}, AIC = {aic_mond:.1f}")

    # 4. Pure MOND asymptote model: log_gobs = d + 0.5*log_gbar
    def mond_asymp_log(x, d):
        return d + 0.5 * x
    r = minimize(lambda p: chi2(p, mond_asymp_log), [-6], method="L-BFGS-B")
    d_mond = r.x[0]
    chi2_ma = r.fun
    aic_ma = aic(chi2_ma, 1)
    print(f"\n  Pure MOND asymptote (1 param):")
    print(f"    log_gobs = {d_mond:.4f} + 0.5·log_gbar")
    print(f"    χ² = {chi2_ma:.2f}, AIC = {aic_ma:.1f}")

    # Summary
    print(f"\n  Model Comparison:")
    print(f"  {'Model':<35s} {'k':<5s} {'χ²':<10s} {'AIC':<10s}")
    print(f"  {'-'*60}")
    best = min([aic_cpx5, aic_c5m, aic_mond, aic_ma])
    for label, k, c, a in [
        ("CPX5 (a + b/log_gbar)", 2, chi2_cpx5, aic_cpx5),
        ("CPX5 + MOND term (a + b/x + c·x)", 3, chi2_c5m, aic_c5m),
        ("McGaugh RAR IF", 1, chi2_mond, aic_mond),
        ("Pure MOND asymptote (d + 0.5·x)", 1, chi2_ma, aic_ma),
    ]:
        print(f"  {label:<35s} {k:<5d} {c:<10.2f} {a:<10.1f} Δ={a-best:.1f}")

    return {
        "cpx5": (a_cpx5, b_cpx5, chi2_cpx5, aic_cpx5),
        "cpx5_mond": (a_c5m, b_c5m, c_c5m, chi2_c5m, aic_c5m),
        "mond": (a0, chi2_mond, aic_mond),
        "mond_asymp": (d_mond, chi2_ma, aic_ma),
    }


def plot_asymptote(x, y, weights, results, label, outdir="analysis"):
    """Plot the fits and extrapolate to low gbar."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fit curves
    ax = axes[0]
    ax.plot(x, y, "ko", ms=3, label="Data")

    x_grid = np.linspace(-16, -8, 500)
    a_cpx5, b_cpx5, _, _ = results["cpx5"]
    a_c5m, b_c5m, c_c5m, _, _ = results["cpx5_mond"]
    a0, _, _ = results["mond"]

    ax.plot(x_grid, cpx5_log(x_grid, a_cpx5, b_cpx5), "b-", lw=2, label="CPX5")
    ax.plot(x_grid, cpx5_mond_log(x_grid, a_c5m, b_c5m, c_c5m), "r--", lw=2,
            label=f"CPX5+MOND (c={c_c5m:.3f})")
    ax.plot(x_grid, mond_mcgaugh_log(x_grid, np.log10(a0)), "g:", lw=2,
            label=f"RAR IF (a₀={a0:.2e})")

    # Extrapolate to MOND regime
    x_ext = np.linspace(-17, -8, 300)
    ax.plot(x_ext, cpx5_mond_log(x_ext, a_c5m, b_c5m, c_c5m), "r:", lw=1, alpha=0.5)
    ax.plot(x_ext, cpx5_log(x_ext, a_cpx5, b_cpx5), "b:", lw=1, alpha=0.5)

    # MOND √gbar line
    ax.plot(x_ext, -5 + 0.5 * (x_ext + 10), "k:", lw=0.5, alpha=0.3, label="√g (slope 1/2)")

    ax.set_xlabel("log gbar (m/s²)")
    ax.set_ylabel("log gobs (m/s²)")
    ax.legend(fontsize=8)
    ax.set_xlim(-17, -8)
    ax.set_ylim(-14, -8)
    ax.set_title(f"MOND asymptote test: {label}")

    # Slope as function of gbar
    ax = axes[1]
    # d(log_gobs)/d(log_gbar) for each model
    slope_cpx5 = -b_cpx5 / x_grid**2
    slope_c5m = -b_c5m / x_grid**2 + c_c5m
    slope_mond = 0.5 * (1 / np.sqrt(1 + 4*10**np.log10(a0)/np.maximum(10**x_grid, 1e-20)))
    # McGaugh RAR IF slope analytically: d/dx 0.5*log(gbar) + ...
    # Numerically:
    eps = 1e-6
    slope_mcg = (mond_mcgaugh_log(x_grid + eps, np.log10(a0))
                 - mond_mcgaugh_log(x_grid - eps, np.log10(a0))) / (2*eps)

    ax.plot(x_grid, slope_cpx5, "b-", lw=2, label="CPX5")
    ax.plot(x_grid, slope_c5m, "r--", lw=2, label=f"CPX5+MOND (c={c_c5m:.3f})")
    ax.plot(x_grid, slope_mcg, "g:", lw=2, label="RAR IF")
    ax.axhline(0.5, color="k", ls=":", lw=0.5, alpha=0.5, label="√g limit")
    ax.axhline(1.0, color="k", ls=":", lw=0.5, alpha=0.3, label="Newtonian")

    ax.set_xlabel("log gbar (m/s²)")
    ax.set_ylabel("d(log gobs) / d(log gbar)")
    ax.legend(fontsize=8)
    ax.set_xlim(-17, -8)
    ax.set_ylim(-0.5, 2)

    plt.tight_layout()
    fname = f"{outdir}/mond_asymptote_{label.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    print(f"  Saved {fname}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs("analysis", exist_ok=True)

    # ── Test 1: SPARC full mass model data ──
    df = parse_mass_models()
    acc = compute_radial_accelerations(df)
    valid = np.isfinite(acc["log_gbar"]) & np.isfinite(acc["log_gobs"]) & (acc["gbar"] > 0)
    x1 = acc["log_gbar"].values[valid]
    y1 = acc["log_gobs"].values[valid]
    w1 = np.ones_like(x1)

    r1 = run_test("SPARC full (3391 pts)", x1, y1, w1)
    plot_asymptote(x1, y1, w1, r1, "SPARC full")

    # ── Test 2: SPARC binned + lensing ──
    x2 = np.concatenate([SPARC_BINNED[:, 0], LENSING[:, 0]])
    y2 = np.concatenate([SPARC_BINNED[:, 1], LENSING[:, 1]])
    err_lens = np.sqrt(LENSING[:, 2]**2 + LENSING[:, 3]**2 + 0.1**2)
    err2 = np.concatenate([SPARC_BINNED[:, 2], err_lens])
    w2 = 1.0 / np.maximum(err2, 1e-10)**2
    w2 = w2 / np.mean(w2)

    r2 = run_test("SPARC binned + Lensing (21 pts)", x2, y2, w2)
    plot_asymptote(x2, y2, w2, r2, "SPARC+lensing")

    print("\nDone.")

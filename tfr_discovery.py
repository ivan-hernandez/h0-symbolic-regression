"""Tully-Fisher Relation: SR discovery + MOND comparison.

The baryonic TFR: M_baryon ∝ V_flat^n. MOND predicts n=4.
We compute V_flat and M_b for each SPARC galaxy and test
whether the data-driven form matches MOND's prediction.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr
from parse_sparc import parse_mass_models
import os, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/tfr"
os.makedirs(OUTDIR, exist_ok=True)

kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0
G_SI = 6.6743e-11
Msun_kg = 1.989e30


def compute_galaxy_properties():
    """Compute V_flat and M_b for each SPARC galaxy."""
    df = parse_mass_models()
    df = df[df["R"] > 0].copy()

    results = []
    for gal in df["ID"].unique():
        sub = df[df["ID"] == gal].sort_values("R")
        if len(sub) < 5:
            continue

        # V_flat: outer velocity (average of last 3 points)
        v_outer = sub["Vobs"].values[-3:]
        V_flat = np.mean(v_outer)
        e_Vflat = np.std(v_outer) / np.sqrt(len(v_outer))

        # V_max: maximum velocity
        V_max = sub["Vobs"].max()

        # Baryonic mass
        R_last_kpc = sub["R"].values[-1]

        # Gas mass enclosed within R_last (proper unit conversion)
        V_gas_sq = (np.abs(sub["Vgas"]) * sub["Vgas"]).mean() * 1e6  # km²/s² → m²/s²
        V_disk_sq = (sub["Vdisk"]**2).mean() * 1e6
        V_bul_sq = (sub["Vbul"]**2).mean() * 1e6
        R_last_m = R_last_kpc * kpc_to_m

        M_gas = V_gas_sq * R_last_m / G_SI / Msun_kg
        M_star = (0.5*V_disk_sq + 0.7*V_bul_sq) * R_last_m / G_SI / Msun_kg
        M_b = M_gas + M_star

        # Gas fraction
        f_gas = M_gas / M_b if M_b > 0 else 0

        # SB
        SB = np.median(sub["SBdisk"] + sub["SBbul"])

        results.append({
            "galaxy": gal, "n_pts": len(sub),
            "V_flat": V_flat, "e_Vflat": e_Vflat,
            "V_max": V_max,
            "M_b": M_b, "M_gas": M_gas, "M_star": M_star,
            "f_gas": f_gas, "SB": SB,
            "D": sub["D"].iloc[0],
        })

    return pd.DataFrame(results)


def fit_tfr(df, outdir=OUTDIR):
    """Fit power law and MOND forms to the TFR."""
    print("=" * 60)
    print("Tully-Fisher Relation: Data-driven Discovery")
    print("=" * 60)

    V = df["V_flat"].values
    M = df["M_b"].values
    eV = df["e_Vflat"].values
    log_V = np.log10(V)
    log_M = np.log10(np.maximum(M, 1e-10))

    # Remove bad fits
    good = np.isfinite(log_V) & np.isfinite(log_M) & (V > 0) & (M > 1e6)
    log_V, log_M, V, M, eV = log_V[good], log_M[good], V[good], M[good], eV[good]
    N = len(V)
    print(f"\n  Valid galaxies: {N}")

    # Weighted fit with minimum scatter floor
    sigma_logM = np.maximum(eV / (V * np.log(10)) * 4.0, 0.05)  # min 0.05 dex scatter

    # 1. Simple power law: log M = a + n * log V
    popt_pl, pcov_pl = curve_fit(lambda x, a, n: a + n*x, log_V, log_M,
                                  sigma=sigma_logM, absolute_sigma=True)
    a_pl, n_pl = popt_pl
    e_a_pl, e_n_pl = np.sqrt(np.diag(pcov_pl))
    pred_pl = a_pl + n_pl * log_V
    chi2_pl = np.sum((log_M - pred_pl)**2 / sigma_logM**2)
    aic_pl = chi2_pl + 2*2

    # 2. MOND prediction: n=4 exactly, fit only log_M = a + 4*log_V
    a_mond = np.mean(log_M - 4*log_V)
    pred_mond = a_mond + 4*log_V
    chi2_mond = np.sum((log_M - pred_mond)**2 / sigma_logM**2)
    aic_mond = chi2_mond + 2*1

    # 3. Broken power law: two segments
    def broken_pl(x, a, n_lo, n_hi, V_break):
        log_V_break = np.log10(V_break)
        mask = x < log_V_break
        pred = np.where(mask, a + n_lo*x, a + n_lo*log_V_break + n_hi*(x - log_V_break))
        return pred

    try:
        popt_bp, _ = curve_fit(broken_pl, log_V, log_M,
                                p0=[a_pl, 4, 3, 80], maxfev=10000)
        pred_bp = broken_pl(log_V, *popt_bp)
        chi2_bp = np.sum((log_M - pred_bp)**2 / sigma_logM**2)
        aic_bp = chi2_bp + 2*4
    except:
        popt_bp = [np.nan]*4
        chi2_bp = np.inf
        aic_bp = np.inf

    # 4. Quadratic: log M = a + n*log_V + c*(log_V)^2
    popt_q, _ = curve_fit(lambda x, a, n, c: a + n*x + c*x**2, log_V, log_M,
                           p0=[a_pl, n_pl, 0], maxfev=10000)
    pred_q = popt_q[0] + popt_q[1]*log_V + popt_q[2]*log_V**2
    chi2_q = np.sum((log_M - pred_q)**2 / sigma_logM**2)
    aic_q = chi2_q + 2*3

    print(f"\n  Model comparison:")
    print(f"  {'Model':<30s} {'k':<5s} {'χ²':<10s} {'AIC':<10s} {'ΔAIC':<10s}")
    print(f"  {'-'*30} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    best = min(aic_pl, aic_mond, aic_bp, aic_q)
    for name, k, c2, aic in [
        (f"Power law (n={n_pl:.2f}±{e_n_pl:.2f})", 2, chi2_pl, aic_pl),
        ("MOND (n=4 fixed)", 1, chi2_mond, aic_mond),
        ("Broken power law", 4, chi2_bp, aic_bp),
        ("Quadratic", 3, chi2_q, aic_q),
    ]:
        marker = " ✓" if aic == best else ""
        print(f"  {name:<30s} {k:<5d} {c2:<10.1f} {aic:<10.1f} {aic-best:+10.1f}{marker}")

    # How many sigma is n=4 from the fitted slope?
    if not np.isnan(n_pl):
        sigma_from_4 = abs(n_pl - 4.0) / e_n_pl
        print(f"\n  Best-fit slope n = {n_pl:.2f} ± {e_n_pl:.2f}")
        print(f"  MOND prediction n = 4.0: {sigma_from_4:.1f}σ {'away' if sigma_from_4>1 else 'consistent'}")

    # Scatter
    scatter = np.std(log_M - pred_pl)
    mond_scatter = np.std(log_M - pred_mond)
    print(f"\n  Scatter (dex):")
    print(f"    Best power law: {scatter:.4f}")
    print(f"    MOND (n=4):     {mond_scatter:.4f}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) TFR
    ax = axes[0]
    sc = ax.scatter(V, M, s=15, alpha=0.6, c=df["f_gas"][good], cmap="viridis",
                    vmin=0, vmax=1, edgecolors="none")
    V_grid = np.logspace(1.3, 2.6, 100)
    ax.plot(V_grid, 10**(a_pl + n_pl*np.log10(V_grid)), "b-", lw=2.5,
            label=f"Power law: n={n_pl:.2f}±{e_n_pl:.2f}")
    ax.plot(V_grid, 10**(a_mond + 4*np.log10(V_grid)), "r--", lw=2,
            label="MOND: n=4.0 (fixed)")
    if not np.isnan(popt_bp[0]):
        ax.plot(V_grid, 10**broken_pl(np.log10(V_grid), *popt_bp), "g:", lw=2,
                label="Broken PL")
    plt.colorbar(sc, ax=ax, label="Gas fraction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("V_flat [km/s]")
    ax.set_ylabel("M_baryon [M_sun]")
    ax.set_title(f"(a) Baryonic TFR ({N} galaxies)")
    ax.legend(fontsize=8, loc="upper left")

    # (b) Residuals
    ax = axes[1]
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.scatter(V, log_M - pred_pl, s=10, alpha=0.5, c="blue", label="Best-fit")
    ax.scatter(V, log_M - pred_mond, s=10, alpha=0.5, c="red", label="MOND n=4")
    ax.set_xscale("log")
    ax.set_xlabel("V_flat [km/s]")
    ax.set_ylabel("Residual (dex)")
    ax.legend(fontsize=8)
    ax.set_title("(b) Residuals")

    # (c) Slope vs gas fraction
    ax = axes[2]
    # Per-galaxy slope estimate: using inner and outer velocities
    slopes = []
    f_gas_vals = []
    for i, gal in enumerate(df["galaxy"].values[good]):
        sub = df[df["galaxy"] == gal]
        if len(sub) < 5:
            continue
        # Rough local slope: log M(R) vs log V(R)
        sub_v = np.log10(sub["V_flat"].values[:5])  # placeholder
        # Use the galaxy's position relative to the global fit
        slopes.append(log_M[i] - (a_pl + n_pl*log_V[i]))
        f_gas_vals.append(df["f_gas"].values[good][i])

    sc = ax.scatter(df["f_gas"][good], log_M - pred_pl, s=15, alpha=0.5, c=log_V, cmap="RdYlBu_r")
    r, p = spearmanr(df["f_gas"][good], log_M - pred_pl)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Gas fraction")
    ax.set_ylabel("Residual from best-fit (dex)")
    ax.set_title(f"(c) Residual vs Gas Fraction (ρ={r:.2f})")
    plt.colorbar(sc, ax=ax, label="log V_flat")

    plt.tight_layout()
    plt.savefig(f"{outdir}/tfr_discovery.pdf", dpi=200)
    plt.savefig(f"{outdir}/tfr_discovery.png", dpi=150)
    print(f"\n  Saved {outdir}/tfr_discovery.png")
    plt.close()

    # Save results
    results = {
        "n_fit": n_pl, "n_err": e_n_pl, "a_fit": a_pl,
        "sigma_from_mond": abs(n_pl-4.0)/e_n_pl if not np.isnan(n_pl) else np.nan,
        "scatter_pl": scatter, "scatter_mond": mond_scatter,
        "aic_pl": aic_pl, "aic_mond": aic_mond,
        "best_model": min([(aic_pl, "power_law"), (aic_mond, "mond"),
                           (aic_bp, "broken_pl"), (aic_q, "quadratic")])[1],
    }
    df_tfr = pd.DataFrame([results])
    df_tfr.to_csv(f"{outdir}/tfr_results.csv", index=False)

    # Also save per-galaxy data
    df[good].to_csv(f"{outdir}/tfr_galaxies.csv", index=False)

    return results


if __name__ == "__main__":
    df = compute_galaxy_properties()
    results = fit_tfr(df)
    print("\nDone.")

"""Phase 3 fix: Mass-weighted EFE proxy.

Replace the 1D nearest-neighbor distance EFE proxy with a physical
external field strength estimate using Tully-Fisher mass estimates.

g_ext ≈ Σ (G · M_dyn_j / r_ij²) from all other SPARC galaxies.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from parse_sparc import parse_mass_models
import os, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/phase3"

kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0
G_SI = 6.6743e-11
Msun_kg = 1.989e30
Mpc_m = 3.0857e22


def mond_simple(gbar, a0):
    return gbar * (1 + np.sqrt(1 + 4*a0/np.maximum(gbar, 1e-20))) / 2


def mond_mcgaugh(gbar, a0):
    return gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20)/a0)), 1e-20)


def cpx5_log(x, a, b):
    return a + b / np.maximum(x, -50)


def load_data():
    df = parse_mass_models()
    df = df[df["R"] > 0].copy()
    Vbar_sq = (np.abs(df["Vgas"])*df["Vgas"]
               + 0.5*df["Vdisk"]**2 + 0.7*df["Vbul"]**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"] * kpc_to_m
    gbar = Vbar_sq * KM_S_TO_M_S**2 / R_m
    gobs = df["Vobs"]**2 * KM_S_TO_M_S**2 / R_m
    valid = (gbar > 1e-13) & (gobs > 0)
    return df[valid].copy(), np.log10(gbar[valid]), np.log10(gobs[valid]), gbar[valid], gobs[valid]


def compute_per_galaxy_masses(df):
    """Estimate M_dyn for each galaxy from V_max² · R_last / G."""
    masses = {}
    for gal in df["ID"].unique():
        sub = df[df["ID"] == gal].sort_values("R")
        V_max = sub["Vobs"].max()
        R_last = sub["R"].iloc[-1]  # kpc
        # M_dyn ≈ V² · R / G (simplified)
        M_dyn = (V_max * 1000)**2 * (R_last * kpc_m) / G_SI / Msun_kg
        D_mpc = sub["D"].iloc[0]  # Mpc
        masses[gal] = {"M_dyn": M_dyn, "D": D_mpc}
    return masses


def compute_external_field(galaxies, masses):
    """g_ext for each galaxy from all other SPARC galaxies."""
    g_ext = {}
    gal_list = list(galaxies)
    for i, gal in enumerate(gal_list):
        if gal not in masses:
            continue
        D_i = masses[gal]["D"]
        total_g = 0.0
        for j, other in enumerate(gal_list):
            if other == gal or other not in masses:
                continue
            D_j = masses[other]["D"]
            M_j = masses[other]["M_dyn"]
            # 3D separation: crude estimate from angular separation?
            # Use projected separation only: d = |D_i - D_j| and assume
            # typical angular separation ~3° → transverse ~0.05 D
            # Better: use the 1D distance as lower bound
            d_proj = abs(D_i - D_j)
            # Add typical transverse component for nearby galaxies
            d_3d = np.sqrt(d_proj**2 + (0.1 * min(D_i, D_j))**2)  # rough
            d_m = d_3d * Mpc_m
            g_j = G_SI * M_j * Msun_kg / d_m**2
            total_g += g_j
        g_ext[gal] = total_g
    return g_ext


def run_mass_weighted_efe(outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Phase 3 Fix: Mass-Weighted EFE Proxy")
    print("=" * 60)

    df, log_gbar, log_gobs, gbar, gobs = load_data()

    # Per-galaxy masses
    masses = compute_per_galaxy_masses(df)
    n_gal = len(masses)
    print(f"\n  Estimated M_dyn for {n_gal} galaxies")
    M_vals = [m["M_dyn"] for m in masses.values()]
    print(f"    M_dyn range: [{min(M_vals):.1e}, {max(M_vals):.1e}] Msun")
    print(f"    Median M_dyn: {np.median(M_vals):.1e} Msun")

    # External field
    g_ext = compute_external_field(df["ID"].unique(), masses)
    print(f"\n  External field range: [{min(g_ext.values()):.2e}, {max(g_ext.values()):.2e}] m/s²")
    print(f"    Median g_ext: {np.median(list(g_ext.values())):.2e} m/s²")
    print(f"    a₀ = 1.2e-10, so typical g_ext/a₀ ≈ {np.median(list(g_ext.values()))/1.2e-10:.1e}")

    # Add to dataframe
    df["M_dyn"] = df["ID"].map({g: masses[g]["M_dyn"] for g in masses})
    df["g_ext"] = df["ID"].map(g_ext)
    df["log_gext"] = np.log10(np.maximum(df["g_ext"], 1e-20))

    # Fit models and compute residuals
    popt_cpx5, _ = curve_fit(cpx5_log, log_gbar, log_gobs, p0=[-17, -70], maxfev=10000)
    df["resid_cpx5"] = log_gobs - cpx5_log(log_gbar, *popt_cpx5)

    popt_sim, _ = curve_fit(mond_simple, gbar, gobs, p0=[1.2e-10], maxfev=10000)
    df["resid_simple"] = log_gobs - np.log10(mond_simple(gbar, *popt_sim))

    popt_mcg, _ = curve_fit(mond_mcgaugh, gbar, gobs, p0=[1.2e-10], maxfev=10000)
    df["resid_mcgaugh"] = log_gobs - np.log10(mond_mcgaugh(gbar, *popt_mcg))

    # Old 1D isolation proxy for comparison
    D_vals = df.groupby("ID")["D"].first().values
    gal_ids = df.groupby("ID")["D"].first().index.values
    isolation = {gal: np.min(np.abs(np.delete(D_vals, i) - D_vals[i]))
                 for i, gal in enumerate(gal_ids)}
    df["log_isol"] = np.log10(np.maximum(df["ID"].map(isolation), 0.1))

    # Correlations: mass-weighted vs 1D isolation
    print(f"\n  Per-point Spearman (g_ext, residual):")
    for resid, label in [("resid_cpx5", "CPX5"), ("resid_simple", "MOND Simple"),
                          ("resid_mcgaugh", "MOND McGaugh")]:
        r_g, p_g = spearmanr(df["log_gext"], df[resid])
        r_i, p_i = spearmanr(df["log_isol"], df[resid])
        print(f"    {label:<18s}: g_ext: ρ={r_g:+.4f} (p={p_g:.2e})  "
              f"1D: ρ={r_i:+.4f} (p={p_i:.2e})")

    print(f"\n  Per-galaxy (mean residual vs g_ext):")
    for resid, label in [("resid_cpx5", "CPX5"), ("resid_simple", "MOND Simple"),
                          ("resid_mcgaugh", "MOND McGaugh")]:
        gal_resid = df.groupby("ID")[resid].mean()
        gal_gext = df.groupby("ID")["log_gext"].first()
        r, p = spearmanr(gal_resid.values, gal_gext.values)
        print(f"    {label:<18s}: ρ={r:+.4f} (p={p:.2e})")

    # ── Figure ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (resid, label) in enumerate([("resid_cpx5", "CPX5"),
                                         ("resid_simple", "MOND Simple"),
                                         ("resid_mcgaugh", "MOND McGaugh")]):
        # Top: per-point vs g_ext
        ax = axes[0, i]
        ax.scatter(df["log_gext"], df[resid], s=1, alpha=0.15, color="steelblue")
        # Binned
        bins = np.linspace(df["log_gext"].min(), df["log_gext"].max(), 15)
        bc, bm, bs = [], [], []
        for j in range(len(bins)-1):
            mask = (df["log_gext"] >= bins[j]) & (df["log_gext"] < bins[j+1])
            if mask.sum() > 10:
                bc.append((bins[j]+bins[j+1])/2)
                bm.append(np.median(df.loc[mask, resid]))
                bs.append(np.std(df.loc[mask, resid])/np.sqrt(mask.sum()))
        ax.errorbar(bc, bm, yerr=bs, fmt="k.-", lw=2, capsize=2)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        r, p = spearmanr(df["log_gext"], df[resid])
        ax.set_xlabel("log g_ext [m/s²]")
        ax.set_ylabel(f"{label} residual (dex)")
        ax.set_title(f"{label}: ρ={r:+.4f} (p={p:.2e})")

        # Bottom: per-galaxy
        ax = axes[1, i]
        gal_resid = df.groupby("ID")[resid].mean()
        gal_gext = df.groupby("ID")["log_gext"].first()
        ax.scatter(gal_gext.values, gal_resid.values, s=15, alpha=0.5, color="steelblue")
        ax.axhline(0, color="k", ls="--", lw=0.5)
        r_g, p_g = spearmanr(gal_resid.values, gal_gext.values)
        ax.set_xlabel("log g_ext [m/s²]")
        ax.set_ylabel(f"Mean {label} residual (dex)")
        ax.set_title(f"Per-galaxy: ρ={r_g:+.4f} (p={p_g:.2e})")

    plt.tight_layout()
    plt.savefig(f"{outdir}/mass_efe_test.pdf", dpi=200)
    plt.savefig(f"{outdir}/mass_efe_test.png", dpi=150)
    print(f"\n  Saved {outdir}/mass_efe_test.png")
    plt.close()

    # Verdict
    print(f"\n  {'='*60}")
    print(f"  VERDICT")
    print(f"  {'='*60}")
    print(f"  Mass-weighted EFE proxy (g_ext from Tully-Fisher M_dyn)")
    print(f"  Typical g_ext/a₀ ≈ {np.median(list(g_ext.values()))/1.2e-10:.1e}")
    if np.median(list(g_ext.values())) < 1.2e-10:
        print(f"  → SPARC galaxies are in the DEEP-MOND EFE regime (g_ext < a₀)")
        print(f"  → If EFE exists, SPARC should show it strongly")
    else:
        print(f"  → SPARC galaxies are in the Newtonian EFE regime (g_ext > a₀)")

    return df


if __name__ == "__main__":
    run_mass_weighted_efe()
    print("\nDone.")

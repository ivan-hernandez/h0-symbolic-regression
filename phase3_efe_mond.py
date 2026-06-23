"""Phase 3: EFE test against MOND residuals (not just CPX5).

The RAR debate flagged that our EFE analysis only tested CPX5 residuals.
This script tests: do isolated galaxies show systematic MOND residual
patterns? If MOND's EFE prediction is real, MOND residuals should
anti-correlate with isolation (isolated galaxies → weaker external field
→ larger deviation from the MOND IF).
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotation_curves"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from parse_sparc import parse_mass_models

OUTDIR = "analysis/phase3"
kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0


def mond_simple(gbar, a0):
    return gbar * (1 + np.sqrt(1 + 4*a0/np.maximum(gbar, 1e-20))) / 2


def mond_mcgaugh(gbar, a0):
    return gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20)/a0)), 1e-20)


def cpx5_log(x, a, b):
    return a + b / np.maximum(x, -50)


def load_and_compute():
    """Load SPARC, compute EFE isolation proxy + all residuals."""
    df = parse_mass_models()
    df = df[df["R"] > 0].copy()

    Vbar_sq = (np.abs(df["Vgas"].values)*df["Vgas"].values
               + 0.5*df["Vdisk"].values**2 + 0.7*df["Vbul"].values**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"].values * kpc_to_m
    gbar = Vbar_sq * KM_S_TO_M_S**2 / R_m
    gobs = df["Vobs"].values**2 * KM_S_TO_M_S**2 / R_m
    valid = (gbar > 1e-13) & (gobs > 0)

    data = pd.DataFrame({
        "ID": df["ID"].values[valid],
        "D": df["D"].values[valid],
        "log_gbar": np.log10(gbar[valid]),
        "log_gobs": np.log10(gobs[valid]),
        "gbar": gbar[valid],
        "gobs": gobs[valid],
    })

    # Global CPX5 fit
    popt_cpx5, _ = curve_fit(cpx5_log, data["log_gbar"], data["log_gobs"],
                              p0=[-17, -70], maxfev=10000)
    data["resid_cpx5"] = data["log_gobs"] - cpx5_log(data["log_gbar"], *popt_cpx5)

    # MOND Simple fit
    popt_simple, _ = curve_fit(mond_simple, data["gbar"], data["gobs"],
                                p0=[1.2e-10], maxfev=10000)
    data["resid_simple"] = data["log_gobs"] - np.log10(mond_simple(data["gbar"], *popt_simple))

    # MOND McGaugh fit
    popt_mcg, _ = curve_fit(mond_mcgaugh, data["gbar"], data["gobs"],
                             p0=[1.2e-10], maxfev=10000)
    data["resid_mcgaugh"] = data["log_gobs"] - np.log10(mond_mcgaugh(data["gbar"], *popt_mcg))

    # Isolation: nearest neighbor distance in Mpc (1D proxy)
    D_vals = data.groupby("ID")["D"].first().values
    gal_ids = data.groupby("ID")["D"].first().index.values
    isolation = {}
    for i, gal in enumerate(gal_ids):
        isolation[gal] = np.min(np.abs(np.delete(D_vals, i) - D_vals[i]))
    data["isolation"] = data["ID"].map(isolation)
    data["log_isol"] = np.log10(np.maximum(data["isolation"], 0.1))

    # Also use SIMBAD 3D distances if available
    try:
        with open("galaxy_coords.json") as f:
            import json
            coords = json.load(f)
        dist_3d = {}
        for gal in gal_ids:
            if gal in coords and coords[gal] is not None:
                d_gal = coords[gal]["D"]
                others = [coords[g]["D"] for g in coords if g != gal and coords[g] is not None]
                if others:
                    dist_3d[gal] = np.min(np.abs(np.array(others) - d_gal))
        data["isolation_3d"] = data["ID"].map(dist_3d)
        data["log_isol_3d"] = np.log10(np.maximum(data["isolation_3d"].fillna(0.5), 0.1))
    except Exception:
        data["log_isol_3d"] = data["log_isol"]

    return data


def test_efe_mond(data, outdir=OUTDIR):
    """Test EFE correlation against MOND residuals."""
    import os
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Phase 3: EFE Test — MOND vs CPX5 Residuals")
    print("=" * 60)

    n_gal = data["ID"].nunique()
    n_pts = len(data)
    print(f"  {n_gal} galaxies, {n_pts} points")

    # Per-point correlations
    print(f"\n  Per-point Spearman (isolation, residual):")
    for resid_col, label in [
        ("resid_cpx5", "CPX5"),
        ("resid_simple", "MOND Simple"),
        ("resid_mcgaugh", "MOND McGaugh"),
    ]:
        r1d, p1d = spearmanr(data["log_isol"], data[resid_col])
        r3d, p3d = spearmanr(data["log_isol_3d"].fillna(0), data[resid_col])
        print(f"    {label:<18s}: 1D: ρ={r1d:+.4f} (p={p1d:.2e})  3D: ρ={r3d:+.4f} (p={p3d:.2e})")

    # Per-galaxy mean residual
    print(f"\n  Per-galaxy mean residual vs isolation:")
    for resid_col, label in [
        ("resid_cpx5", "CPX5"),
        ("resid_simple", "MOND Simple"),
        ("resid_mcgaugh", "MOND McGaugh"),
    ]:
        gal_resid = data.groupby("ID")[resid_col].mean()
        gal_isol = data.groupby("ID")["isolation"].first()
        r, p = spearmanr(gal_resid.values, gal_isol.values)
        print(f"    {label:<18s}: ρ={r:+.4f} (p={p:.2e})")

    # MOND EFE prediction: isolated galaxies → more negative residual
    # (MOND overpredicts gobs when EFE is present in non-isolated galaxies)
    # So EFE predicts NEGATIVE correlation: more isolated → less EFE → higher gobs → positive resid
    # Let's check the sign
    isolated = data[data["isolation"] > 2.0]  # >2 Mpc from nearest
    non_isolated = data[data["isolation"] < 2.0]
    print(f"\n  Mean residuals by isolation:")
    for resid_col, label in [
        ("resid_cpx5", "CPX5"),
        ("resid_simple", "MOND Simple"),
        ("resid_mcgaugh", "MOND McGaugh"),
    ]:
        mean_iso = isolated[resid_col].mean()
        mean_non = non_isolated[resid_col].mean()
        diff = mean_iso - mean_non
        print(f"    {label:<18s}: iso={mean_iso:+.5f}  non-iso={mean_non:+.5f}  "
              f"Δ={diff:+.5f}  {'(EFE WARNING)' if diff > 0.01 else '(consistent)'}")

    # ── Figure ───────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (resid_col, label) in enumerate([
        ("resid_cpx5", "CPX5"),
        ("resid_simple", "MOND Simple"),
        ("resid_mcgaugh", "MOND McGaugh"),
    ]):
        # Top row: scatter
        ax = axes[0, i]
        ax.scatter(data["log_isol"], data[resid_col], s=1, alpha=0.2,
                   c="steelblue" if "cpx5" in resid_col else
                   ("darkorange" if "simple" in resid_col else "red"))
        # Binned
        bins = np.linspace(data["log_isol"].min(), data["log_isol"].max(), 15)
        bc, bm, bs = [], [], []
        for j in range(len(bins)-1):
            mask = (data["log_isol"] >= bins[j]) & (data["log_isol"] < bins[j+1])
            if mask.sum() > 10:
                bc.append((bins[j]+bins[j+1])/2)
                bm.append(np.median(data.loc[mask, resid_col]))
                bs.append(np.std(data.loc[mask, resid_col])/np.sqrt(mask.sum()))
        ax.errorbar(bc, bm, yerr=bs, fmt="k.-", lw=2, capsize=2)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        r, p = spearmanr(data["log_isol"], data[resid_col])
        ax.set_xlabel("log isolation (Mpc)")
        ax.set_ylabel(f"{label} residual (dex)")
        ax.set_title(f"{label}: ρ={r:+.4f} (p={p:.2e})")

        # Bottom row: per-galaxy
        ax = axes[1, i]
        gal_resid = data.groupby("ID")[resid_col].mean()
        gal_isol = data.groupby("ID")["isolation"].first()
        ax.scatter(np.log10(np.maximum(gal_isol.values, 0.1)), gal_resid.values,
                   s=10, alpha=0.5, c="steelblue" if "cpx5" in resid_col else
                   ("darkorange" if "simple" in resid_col else "red"))
        ax.axhline(0, color="k", ls="--", lw=0.5)
        r_gal, p_gal = spearmanr(gal_resid.values, gal_isol.values)
        ax.set_xlabel("log isolation (Mpc)")
        ax.set_ylabel(f"Mean {label} residual (dex)")
        ax.set_title(f"Per-galaxy: ρ={r_gal:+.4f} (p={p_gal:.2e})")

    plt.tight_layout()
    plt.savefig(f"{outdir}/efe_mond_test.pdf", dpi=200)
    plt.savefig(f"{outdir}/efe_mond_test.png", dpi=150)
    print(f"\n  Saved {outdir}/efe_mond_test.png")
    plt.close()

    # Verdict
    print(f"\n  {'='*60}")
    print(f"  VERDICT")
    print(f"  {'='*60}")
    all_r = [spearmanr(data["log_isol"], data[c])[0]
             for c in ["resid_cpx5", "resid_simple", "resid_mcgaugh"]]
    all_signs = ["+" if r > 0 else "-" for r in all_r]

    if all(r > 0 for r in all_r):
        print(f"  All three models show POSITIVE correlation (all ρ > 0)")
        print(f"  Isolated galaxies have SLIGHTLY HIGHER g_obs than predicted.")
        print(f"  This is OPPOSITE to MOND's EFE prediction.")
        print(f"  → EFE NOT DETECTED in MOND residuals either.")
    elif all(r < 0 for r in all_r):
        print(f"  All three models show NEGATIVE correlation")
        print(f"  → EFE signal detected — but universal, not MOND-specific.")
    else:
        print(f"  Mixed signs. No consistent EFE signal.")
        print(f"  → EFE test inconclusive.")

    return data


if __name__ == "__main__":
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    data = load_and_compute()
    test_efe_mond(data)
    print("\nDone.")

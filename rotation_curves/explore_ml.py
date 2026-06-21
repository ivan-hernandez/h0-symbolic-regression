"""Deep dive on Challenge 4: M/L variation and a₀.

Three approaches:
1. Per-galaxy M/L optimization — find each galaxy's best M/L from RAR
2. Global M/L sweep — find M/L that minimizes RAR scatter
3. Check if a₀ converges when M/L is optimized
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from parse_sparc import parse_mass_models


# ── Physics ───────────────────────────────────────────────────────────────────
def compute_gbar(Vgas, Vdisk, Vbul, R, Ud, Ub):
    Vbar_sq = (np.abs(Vgas) * Vgas + Ud * Vdisk**2 + Ub * Vbul**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    kpc_to_m = 3.0857e19
    R_m = R * kpc_to_m
    return Vbar_sq * 1e6 / R_m  # m/s²

def compute_gobs(Vobs, R):
    kpc_to_m = 3.0857e19
    R_m = R * kpc_to_m
    return Vobs**2 * 1e6 / R_m

def mond_simple(gbar, a0):
    return gbar * (1 + np.sqrt(1 + 4 * a0 / np.maximum(gbar, 1e-20))) / 2


# ── 1. Per-galaxy M/L optimization ───────────────────────────────────────────
def optimize_per_galaxy(df, a0_ref=1.2e-10):
    """For each galaxy, find M/L that minimizes offset from MOND prediction."""
    print("=" * 60)
    print("Per-galaxy M/L optimization")
    print("=" * 60)

    df = df[df["R"] > 0].copy()
    results = []
    for gal in df["ID"].unique():
        sub = df[df["ID"] == gal]
        Vgas = sub["Vgas"].values
        Vdisk = sub["Vdisk"].values
        Vbul = sub["Vbul"].values
        R = sub["R"].values
        Vobs = sub["Vobs"].values
        gobs = compute_gobs(Vobs, R)

        def cost(params):
            Ud, Ub = params
            if Ud < 0.1 or Ud > 3 or Ub < 0.1 or Ub > 3:
                return 1e10
            gbar = compute_gbar(Vgas, Vdisk, Vbul, R, Ud, Ub)
            valid = (gbar > 1e-13) & (gobs > 0)
            if valid.sum() < 3:
                return 1e10
            gbar_v = gbar[valid]
            gobs_v = gobs[valid]
            pred = mond_simple(gbar_v, a0_ref)
            return np.mean((np.log10(gobs_v) - np.log10(pred))**2)

        res = minimize(cost, [0.5, 0.7], method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
        Ud_best, Ub_best = res.x
        rms = res.fun

        # Compute a₀ that best fits this galaxy with its optimal M/L
        gbar_opt = compute_gbar(Vgas, Vdisk, Vbul, R, Ud_best, Ub_best)
        valid = (gbar_opt > 1e-13) & (gobs > 0)
        if valid.sum() >= 5:
            try:
                popt, _ = curve_fit(mond_simple, gbar_opt[valid], gobs[valid],
                                     p0=[a0_ref], maxfev=2000)
                a0_gal = popt[0]
            except Exception:
                a0_gal = np.nan
        else:
            a0_gal = np.nan

        results.append({
            "galaxy": gal,
            "Ud_best": Ud_best, "Ub_best": Ub_best,
            "n_pts": valid.sum(),
            "rms_RAR": rms * 100,  # in % (dex×100 ≈ %)
            "a0_gal": a0_gal,
        })

    df_r = pd.DataFrame(results)
    df_r = df_r.sort_values("galaxy")
    print(f"  Optimized {len(df_r)} galaxies")
    print(f"  Ud range: [{df_r['Ud_best'].min():.2f}, {df_r['Ud_best'].max():.2f}]")
    print(f"  Ub range: [{df_r['Ub_best'].min():.2f}, {df_r['Ub_best'].max():.2f}]")
    print(f"  Ud mean ± std: {df_r['Ud_best'].mean():.3f} ± {df_r['Ud_best'].std():.3f}")
    print(f"  Ub mean ± std: {df_r['Ub_best'].mean():.3f} ± {df_r['Ub_best'].std():.3f}")
    print(f"  a₀ mean ± std: {df_r['a0_gal'].mean():.3e} ± {df_r['a0_gal'].std():.3e}")
    print(f"  a₀ median: {df_r['a0_gal'].median():.3e}")

    # Compare SPARC default (0.5, 0.7) vs optimized
    default_rms = []
    for gal in df["ID"].unique():
        sub = df[df["ID"] == gal]
        gbar_def = compute_gbar(sub["Vgas"].values, sub["Vdisk"].values,
                                 sub["Vbul"].values, sub["R"].values, 0.5, 0.7)
        gobs = compute_gobs(sub["Vobs"].values, sub["R"].values)
        valid = (gbar_def > 1e-13) & (gobs > 0)
        if valid.sum() < 3:
            continue
        pred = mond_simple(gbar_def[valid], a0_ref)
        rms = np.mean((np.log10(gobs[valid]) - np.log10(pred))**2)
        default_rms.append(rms)
    print(f"  Default M/L RMS: {np.mean(default_rms)*100:.3f}%")
    print(f"  Optimized M/L RMS: {df_r['rms_RAR'].mean():.3f}%")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(df_r["Ud_best"], bins=30, alpha=0.7)
    axes[0, 0].axvline(0.5, color="k", ls="--", label="SPARC default")
    axes[0, 0].set_xlabel("Optimal ϒ_disk")
    axes[0, 0].set_ylabel("N galaxies")
    axes[0, 0].legend()

    axes[0, 1].hist(df_r["Ub_best"], bins=30, alpha=0.7)
    axes[0, 1].axvline(0.7, color="k", ls="--", label="SPARC default")
    axes[0, 1].set_xlabel("Optimal ϒ_bul")
    axes[0, 1].set_ylabel("N galaxies")
    axes[0, 1].legend()

    axes[1, 0].hist(df_r["a0_gal"][df_r["a0_gal"].notna()], bins=30, alpha=0.7)
    axes[1, 0].axvline(1.2e-10, color="k", ls="--", label="canonical a₀")
    axes[1, 0].axvline(df_r["a0_gal"].median(), color="r", ls="-", label="median")
    axes[1, 0].set_xlabel("a₀ per galaxy (m/s²)")
    axes[1, 0].set_ylabel("N galaxies")
    axes[1, 0].legend()

    axes[1, 1].scatter(df_r["Ud_best"], df_r["a0_gal"], alpha=0.5, s=20)
    axes[1, 1].set_xlabel("ϒ_disk")
    axes[1, 1].set_ylabel("a₀ (m/s²)")
    axes[1, 1].axhline(1.2e-10, color="k", ls="--")

    plt.tight_layout()
    plt.savefig("analysis/per_galaxy_ml.png", dpi=150)
    print("Saved analysis/per_galaxy_ml.png")
    plt.close()

    df_r.to_csv("analysis/per_galaxy_ml.csv", index=False)
    print("Saved analysis/per_galaxy_ml.csv")
    return df_r


# ── 2. Global M/L sweep ─────────────────────────────────────────────────────
def global_ml_sweep(df, a0_ref=1.2e-10):
    """Sweep global M/L to find combination that minimizes RAR scatter."""
    print("\n" + "=" * 60)
    print("Global M/L sweep")
    print("=" * 60)

    df = df[df["R"] > 0].copy()
    Ud_grid = np.linspace(0.2, 1.5, 14)
    Ub_grid = np.linspace(0.2, 1.5, 14)

    results = []
    for Ud in Ud_grid:
        for Ub in Ub_grid:
            gbar = compute_gbar(df["Vgas"].values, df["Vdisk"].values,
                                 df["Vbul"].values, df["R"].values, Ud, Ub)
            gobs = compute_gobs(df["Vobs"].values, df["R"].values)
            valid = (gbar > 1e-13) & (gobs > 0)
            gbar_v, gobs_v = gbar[valid], gobs[valid]
            pred = mond_simple(gbar_v, a0_ref)
            rms = np.sqrt(np.mean((np.log10(gobs_v) - np.log10(pred))**2))
            results.append({"Ud": Ud, "Ub": Ub, "rms_RAR": rms})

    df_r = pd.DataFrame(results)
    best = df_r.loc[df_r["rms_RAR"].idxmin()]
    print(f"  Best global M/L: Ud={best['Ud']:.2f}, Ub={best['Ub']:.2f}")
    print(f"  RAR RMS at best: {best['rms_RAR']*100:.3f}%")
    print(f"  SPARC default RMS: {df_r[(df_r['Ud']==0.5)&(df_r['Ub']==0.7)]['rms_RAR'].values[0]*100:.3f}%")

    # Fit a₀ at best M/L
    gbar_best = compute_gbar(df["Vgas"].values, df["Vdisk"].values,
                              df["Vbul"].values, df["R"].values,
                              best["Ud"], best["Ub"])
    gobs = compute_gobs(df["Vobs"].values, df["R"].values)
    valid = (gbar_best > 1e-13) & (gobs > 0)
    try:
        popt, _ = curve_fit(mond_simple, gbar_best[valid], gobs[valid],
                             p0=[a0_ref], maxfev=5000)
        print(f"  a₀ at best M/L: {popt[0]:.3e}")
    except Exception as e:
        print(f"  a₀ fit failed: {e}")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = df_r.pivot(index="Ub", columns="Ud", values="rms_RAR")
    im = ax.contourf(pivot.columns, pivot.index, pivot.values * 100,
                      levels=20, cmap="viridis")
    ax.plot(0.5, 0.7, "r*", markersize=15, label="SPARC default")
    ax.plot(best["Ud"], best["Ub"], "rX", markersize=15, label="Best M/L")
    plt.colorbar(im, ax=ax, label="RAR RMS (dex×100 ≈ %)")
    ax.set_xlabel("ϒ_disk")
    ax.set_ylabel("ϒ_bul")
    ax.legend()
    plt.tight_layout()
    plt.savefig("analysis/global_ml_heatmap.png", dpi=150)
    print("Saved analysis/global_ml_heatmap.png")
    plt.close()

    df_r.to_csv("analysis/global_ml_sweep.csv", index=False)
    print("Saved analysis/global_ml_sweep.csv")
    return df_r, best


# ── 3. a₀ vs M/L trend ─────────────────────────────────────────────────────
def a0_vs_ml_trend(df):
    """Trace how a₀ varies as we move from low to high M/L."""
    print("\n" + "=" * 60)
    print("a₀ vs M/L trend")
    print("=" * 60)

    # Use M/L pairs along a diagonal from (0.3, 0.3) to (1.5, 1.5)
    ml_vals = np.linspace(0.3, 2.0, 18)
    results = []
    for ml in ml_vals:
        gbar = compute_gbar(df["Vgas"].values, df["Vdisk"].values,
                             df["Vbul"].values, df["R"].values, ml, ml)
        gobs = compute_gobs(df["Vobs"].values, df["R"].values)
        valid = (gbar > 1e-13) & (gobs > 0)
        if valid.sum() < 100:
            continue
        try:
            popt, pcov = curve_fit(mond_simple, gbar[valid], gobs[valid],
                                    p0=[1.2e-10], maxfev=5000)
            a0 = popt[0]
            perr = np.sqrt(np.diag(pcov))[0]
        except Exception:
            a0, perr = np.nan, np.nan
        results.append({"M_L": ml, "a0": a0, "a0_err": perr})

    df_r = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df_r["M_L"], df_r["a0"], yerr=df_r["a0_err"],
                 fmt="o-", capsize=3)
    ax.axhline(1.2e-10, color="k", ls="--", label="canonical a₀")
    ax.axvline(0.5, color="gray", ls=":", label="SPARC ϒ_disk")
    ax.axvline(0.7, color="gray", ls=":")
    ax.set_xlabel("M/L (diagonal)")
    ax.set_ylabel("a₀ (m/s²)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("analysis/a0_vs_ml.png", dpi=150)
    print("Saved analysis/a0_vs_ml.png")
    plt.close()

    df_r.to_csv("analysis/a0_vs_ml.csv", index=False)
    print("Saved analysis/a0_vs_ml.csv")

    # Key insight: interpolation to canonical a₀
    from scipy.interpolate import interp1d
    valid = df_r["a0"].notna() & (df_r["a0"] > 0)
    if valid.sum() > 3:
        f = interp1d(df_r.loc[valid, "a0"].values, df_r.loc[valid, "M_L"].values,
                      bounds_error=False, fill_value=np.nan)
        ml_for_canon = f(1.2e-10)
        print(f"  M/L needed for a₀=1.2e-10: ϒ ≈ {ml_for_canon:.2f}")

    return df_r


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("M/L Deep Dive — Challenge 4")
    print("=" * 60)

    df = parse_mass_models()

    # 1. Per-galaxy M/L
    per_gal = optimize_per_galaxy(df)

    # 2. Global sweep
    sweep, best = global_ml_sweep(df)

    # 3. a₀ trend
    trend = a0_vs_ml_trend(df)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Per-galaxy: Ud={per_gal['Ud_best'].mean():.2f}±{per_gal['Ud_best'].std():.2f}, "
          f"Ub={per_gal['Ub_best'].mean():.2f}±{per_gal['Ub_best'].std():.2f}")
    print(f"  Global sweep best: Ud={best['Ud']:.2f}, Ub={best['Ub']:.2f}")
    print(f"  a₀ per galaxy: median={per_gal['a0_gal'].median():.3e}, "
          f"mean={per_gal['a0_gal'].mean():.3e}±{per_gal['a0_gal'].std():.3e}")
    print(f"  RMS improvement: default {sweep[(sweep['Ud']==0.5)&(sweep['Ub']==0.7)]['rms_RAR'].values[0]*100:.2f}% → "
          f"best {best['rms_RAR']*100:.2f}%")
    print("\nDone.")

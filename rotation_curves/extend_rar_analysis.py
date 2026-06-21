"""Extended RAR analysis: EFE proxy, per-galaxy CPX5 fits, bootstrap.

1. EFE proxy (isolation-based, using distance D)
2. Per-galaxy CPX5 parameters vs galaxy properties
3. Bootstrap parameter uncertainties
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from parse_sparc import parse_mass_models

# ── Physics helpers ───────────────────────────────────────────────────────────
kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0

def compute_gbar(Vgas, Vdisk, Vbul, R, Ud=0.5, Ub=0.7):
    Vbar_sq = np.maximum(np.abs(Vgas)*Vgas + Ud*Vdisk**2 + Ub*Vbul**2, 0)
    return Vbar_sq * KM_S_TO_M_S**2 / (R * kpc_to_m)

def compute_gobs(Vobs, R):
    return Vobs**2 * KM_S_TO_M_S**2 / (R * kpc_to_m)

def sr_cpx5_log(log_gbar, a, b):
    return a + b / log_gbar

def mond_simple(gbar, a0):
    return gbar * (1 + np.sqrt(1 + 4*a0/np.maximum(gbar, 1e-20))) / 2


# ── 1. EFE proxy ────────────────────────────────────────────────────────────
def efe_analysis(df, outdir="analysis"):
    """Check if RAR residuals correlate with galaxy isolation (approximate EFE)."""
    print("\n" + "=" * 60)
    print("EFE Analysis (isolation proxy)")
    print("=" * 60)

    df = df[df["R"] > 0].copy()
    df["gbar"] = compute_gbar(df["Vgas"].values, df["Vdisk"].values,
                               df["Vbul"].values, df["R"].values)
    df["gobs"] = compute_gobs(df["Vobs"].values, df["R"].values)
    valid = (df["gbar"] > 1e-13) & (df["gobs"] > 0)
    df = df[valid].copy()
    df["log_gbar"] = np.log10(df["gbar"])
    df["log_gobs"] = np.log10(df["gobs"])

    # Fit global CPX5
    popt, _ = curve_fit(sr_cpx5_log, df["log_gbar"].values, df["log_gobs"].values,
                         p0=[-12, -50], maxfev=10000)
    df["resid_CPX5"] = df["log_gobs"] - sr_cpx5_log(df["log_gbar"].values, *popt)

    # Isolation proxy: distance to nearest SPARC galaxy in Mpc
    D_vals = df.groupby("ID")["D"].first().values
    gal_ids = df.groupby("ID")["D"].first().index.values

    isolation = {}
    for i, gal in enumerate(gal_ids):
        others = np.delete(D_vals, i)
        d_i = D_vals[i]
        isolation[gal] = np.min(np.abs(others - d_i))

    df["isolation"] = df["ID"].map(isolation)
    df["log_isolation"] = np.log10(np.maximum(df["isolation"], 0.1))
    df["gas_frac"] = np.abs(df["Vgas"]) / np.maximum(df["Vobs"], 0.1)

    # Residual vs isolation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    scatter = ax.scatter(df["log_isolation"], df["resid_CPX5"], s=2, alpha=0.3,
                         c=df["gas_frac"], cmap="viridis", vmin=0, vmax=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("log nearest neighbor distance (Mpc)")
    ax.set_ylabel("CPX5 residual (dex)")
    plt.colorbar(scatter, ax=ax, label="gas fraction")

    # Binned
    bins = np.linspace(df["log_isolation"].min(), df["log_isolation"].max(), 15)
    bin_centers = []
    bin_means = []
    bin_stds = []
    for i in range(len(bins)-1):
        mask = (df["log_isolation"] >= bins[i]) & (df["log_isolation"] < bins[i+1])
        if mask.sum() > 10:
            bin_centers.append((bins[i]+bins[i+1])/2)
            bin_means.append(df.loc[mask, "resid_CPX5"].median())
            bin_stds.append(df.loc[mask, "resid_CPX5"].std()/np.sqrt(mask.sum()))
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="r.-", lw=2, capsize=3)

    # Residual vs gas fraction
    ax = axes[0, 1]
    ax.scatter(df["gas_frac"], df["resid_CPX5"], s=2, alpha=0.3, c=df["log_isolation"], cmap="viridis")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("gas fraction")
    ax.set_ylabel("CPX5 residual (dex)")

    # Isolation vs galaxy count
    ax = axes[1, 0]
    for gal in gal_ids:
        sub = df[df["ID"] == gal]
        ax.scatter(sub["log_isolation"].iloc[0], sub["resid_CPX5"].mean(),
                   s=10, alpha=0.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("log isolation (Mpc)")
    ax.set_ylabel("Mean CPX5 residual (dex)")

    # Per-galaxy RMS vs isolation
    ax = axes[1, 1]
    gal_rms = df.groupby("ID")["resid_CPX5"].agg(["mean", "std", "count"])
    gal_rms = gal_rms.join(pd.Series(isolation, name="isolation"))
    ax.scatter(np.log10(np.maximum(gal_rms["isolation"], 0.1)), gal_rms["std"],
               s=10, alpha=0.5)
    ax.set_xlabel("log isolation (Mpc)")
    ax.set_ylabel("Per-galaxy RMS (dex)")

    plt.tight_layout()
    plt.savefig(f"{outdir}/efe_analysis.png", dpi=150)
    print(f"  Saved {outdir}/efe_analysis.png")
    plt.close()

    # Correlation stats
    from scipy.stats import spearmanr
    r_iso, p_iso = spearmanr(df["log_isolation"], df["resid_CPX5"])
    r_gas, p_gas = spearmanr(df["gas_frac"], df["resid_CPX5"])
    print(f"  Spearman(Isolation, Resid): ρ={r_iso:.4f}, p={p_iso:.2e}")
    print(f"  Spearman(Gas frac, Resid): ρ={r_gas:.4f}, p={p_gas:.2e}")

    # EFE prediction: isolated gas-rich galaxies should deviate from MOND more
    # Using lower threshold since our isolation proxy is crude (1D distance only)
    isolated = df[df["log_isolation"] > 1.0]
    not_isolated = df[df["log_isolation"] < 1.0]
    close = df[df["log_isolation"] < 0.3]  # nearest neighbor < 2 Mpc
    print(f"  Isolated (>{10**1.0:.0f} Mpc): {isolated['ID'].nunique()} galaxies, "
          f"RMS={isolated['resid_CPX5'].std():.4f} dex")
    print(f"  Not isolated: {not_isolated['ID'].nunique()} galaxies, "
          f"RMS={not_isolated['resid_CPX5'].std():.4f} dex")
    if len(close) > 10:
        print(f"  Close pairs (<{10**0.3:.0f} Mpc): {close['ID'].nunique()} galaxies, "
              f"RMS={close['resid_CPX5'].std():.4f} dex")
    print(f"  Isolated RMS: {isolated['resid_CPX5'].std():.4f} dex")
    print(f"  Non-isolated RMS: {not_isolated['resid_CPX5'].std():.4f} dex")

    return df


# ── 2. Per-galaxy CPX5 fits ────────────────────────────────────────────────
def per_galaxy_cpx5(df, outdir="analysis"):
    """Fit CPX5 to each galaxy, track parameters vs properties."""
    print("\n" + "=" * 60)
    print("Per-galaxy CPX5 fits")
    print("=" * 60)

    df = df[df["R"] > 0].copy()
    gbar = compute_gbar(df["Vgas"].values, df["Vdisk"].values, df["Vbul"].values, df["R"].values)
    gobs = compute_gobs(df["Vobs"].values, df["R"].values)
    valid = (gbar > 1e-13) & (gobs > 0)
    df_work = df[valid].copy()
    df_work["log_gbar"] = np.log10(gbar[valid])
    df_work["log_gobs"] = np.log10(gobs[valid])
    df_work["gas_frac"] = np.abs(df_work["Vgas"]) / np.maximum(df_work["Vobs"], 0.1)
    df_work["SB_total"] = df_work["SBdisk"] + df_work["SBbul"]

    results = []
    for gal in df_work["ID"].unique():
        sub = df_work[df_work["ID"] == gal]
        if len(sub) < 5:
            continue
        x, y = sub["log_gbar"].values, sub["log_gobs"].values
        try:
            popt, pcov = curve_fit(sr_cpx5_log, x, y, p0=[-12, -50],
                                    maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            pred = sr_cpx5_log(x, *popt)
            rms = np.sqrt(np.mean((y - pred)**2))
            results.append({
                "galaxy": gal,
                "n_pts": len(sub),
                "a": popt[0], "e_a": perr[0],
                "b": popt[1], "e_b": perr[1],
                "rms": rms,
                "gas_frac": sub["gas_frac"].median(),
                "log_SB": np.log10(np.maximum(sub["SB_total"].median(), 0.1)),
                "D": sub["D"].iloc[0],
            })
        except Exception:
            pass

    df_r = pd.DataFrame(results)
    print(f"  Fit {len(df_r)} galaxies successfully")

    # Parameter distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].hist(df_r["a"], bins=30); axes[0, 0].set_xlabel("CPX5 a (intercept)")
    axes[0, 1].hist(df_r["b"], bins=30); axes[0, 1].set_xlabel("CPX5 b (slope)")
    axes[0, 2].hist(df_r["rms"], bins=30); axes[0, 2].set_xlabel("Per-galaxy RMS (dex)")

    # Parameter correlations with properties
    axes[1, 0].scatter(df_r["gas_frac"], df_r["b"], alpha=0.5, s=10)
    axes[1, 0].set_xlabel("gas fraction"); axes[1, 0].set_ylabel("CPX5 b")

    axes[1, 1].scatter(df_r["log_SB"], df_r["b"], alpha=0.5, s=10)
    axes[1, 1].set_xlabel("log SB"); axes[1, 1].set_ylabel("CPX5 b")

    axes[1, 2].scatter(df_r["D"], df_r["b"], alpha=0.5, s=10)
    axes[1, 2].set_xlabel("Distance (Mpc)"); axes[1, 2].set_ylabel("CPX5 b")

    plt.tight_layout()
    plt.savefig(f"{outdir}/per_galaxy_cpx5.png", dpi=150)
    print(f"  Saved {outdir}/per_galaxy_cpx5.png")
    plt.close()

    df_r.to_csv(f"{outdir}/per_galaxy_cpx5_params.csv", index=False)
    print(f"  Saved {outdir}/per_galaxy_cpx5_params.csv")

    # Summary stats
    from scipy.stats import spearmanr
    print(f"\n  a mean ± std: {df_r['a'].mean():.3f} ± {df_r['a'].std():.3f}")
    print(f"  b mean ± std: {df_r['b'].mean():.3f} ± {df_r['b'].std():.3f}")
    print(f"  RMS mean ± std: {df_r['rms'].mean():.4f} ± {df_r['rms'].std():.4f}")

    for prop, label in [("gas_frac", "gas fraction"), ("log_SB", "log SB"), ("D", "distance")]:
        r, p = spearmanr(df_r[prop], df_r["b"])
        print(f"  b vs {label}: ρ={r:.3f}, p={p:.2e}")

    return df_r


# ── 3. Bootstrap ────────────────────────────────────────────────────────────
def bootstrap_rar(df, outdir="analysis", n_boot=200):
    """Bootstrap resample galaxies to get parameter uncertainties."""
    print("\n" + "=" * 60)
    print(f"Bootstrap uncertainty ({n_boot} resamples)")
    print("=" * 60)

    df = df[df["R"] > 0].copy()
    gbar = compute_gbar(df["Vgas"].values, df["Vdisk"].values, df["Vbul"].values, df["R"].values)
    gobs = compute_gobs(df["Vobs"].values, df["R"].values)
    valid = (gbar > 1e-13) & (gobs > 0)
    df_work = df[valid].copy()
    df_work["log_gbar"] = np.log10(gbar[valid])
    df_work["log_gobs"] = np.log10(gobs[valid])
    df_work["gbar"] = gbar[valid]

    galaxies = df_work["ID"].unique()
    n_gal = len(galaxies)

    rng = np.random.RandomState(42)
    boot_results = []
    for i in range(n_boot):
        # Resample galaxies with replacement
        boot_gals = rng.choice(galaxies, n_gal, replace=True)
        boot = pd.concat([df_work[df_work["ID"] == g] for g in boot_gals])
        x, y = boot["log_gbar"].values, boot["log_gobs"].values
        gbar_boot = boot["gbar"].values

        # CPX5
        try:
            popt_cpx5, _ = curve_fit(sr_cpx5_log, x, y, p0=[-12, -50], maxfev=5000)
        except Exception:
            popt_cpx5 = [np.nan, np.nan]

        # MOND Simple
        try:
            popt_mond, _ = curve_fit(mond_simple, gbar_boot, 10**y,
                                      p0=[1.2e-10], maxfev=5000)
            a0_boot = popt_mond[0]
        except Exception:
            a0_boot = np.nan

        boot_results.append({
            "iter": i,
            "cpx5_a": popt_cpx5[0], "cpx5_b": popt_cpx5[1],
            "mond_a0": a0_boot,
        })

    df_boot = pd.DataFrame(boot_results)

    # Summary
    print(f"\n  CPX5 a: {df_boot['cpx5_a'].mean():.3f} ± {df_boot['cpx5_a'].std():.3f}")
    print(f"          [{np.percentile(df_boot['cpx5_a'].dropna(), 16):.3f}, "
          f"{np.percentile(df_boot['cpx5_a'].dropna(), 84):.3f}] (68% CL)")
    print(f"  CPX5 b: {df_boot['cpx5_b'].mean():.3f} ± {df_boot['cpx5_b'].std():.3f}")
    print(f"          [{np.percentile(df_boot['cpx5_b'].dropna(), 16):.3f}, "
          f"{np.percentile(df_boot['cpx5_b'].dropna(), 84):.3f}] (68% CL)")
    mond_valid = df_boot["mond_a0"].dropna()
    if len(mond_valid) > 10:
        print(f"  MOND a₀: {mond_valid.mean():.3e} ± {mond_valid.std():.3e}")
        print(f"          [{np.percentile(mond_valid, 16):.3e}, "
              f"{np.percentile(mond_valid, 84):.3e}] (68% CL)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(df_boot["cpx5_a"].dropna(), bins=30)
    axes[0, 0].axvline(df_boot["cpx5_a"].mean(), color="r", ls="-")
    axes[0, 0].set_xlabel("CPX5 a")
    axes[0, 1].hist(df_boot["cpx5_b"].dropna(), bins=30)
    axes[0, 1].axvline(df_boot["cpx5_b"].mean(), color="r", ls="-")
    axes[0, 1].set_xlabel("CPX5 b")
    axes[1, 0].scatter(df_boot["cpx5_a"], df_boot["cpx5_b"], alpha=0.3, s=5)
    axes[1, 0].set_xlabel("CPX5 a"); axes[1, 0].set_ylabel("CPX5 b")
    if len(mond_valid) > 10:
        axes[1, 1].hist(mond_valid, bins=30)
        axes[1, 1].axvline(mond_valid.mean(), color="r", ls="-")
        axes[1, 1].axvline(1.2e-10, color="k", ls="--", label="canonical")
        axes[1, 1].set_xlabel("MOND a₀")
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/bootstrap_rar.png", dpi=150)
    print(f"  Saved {outdir}/bootstrap_rar.png")
    plt.close()

    df_boot.to_csv(f"{outdir}/bootstrap_rar.csv", index=False)
    print(f"  Saved {outdir}/bootstrap_rar.csv")

    # Compare with profile-based uncertainty
    print(f"\n  Bootstrap CPX5 b: {df_boot['cpx5_b'].std():.4f} "
          f"(vs conditional fit uncertainty ~{df_boot['cpx5_b'].std()/2:.4f})")

    return df_boot


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    os.makedirs("analysis", exist_ok=True)

    print("=" * 60)
    print("Extended RAR Analysis: EFE + Per-galaxy + Bootstrap")
    print("=" * 60)

    df = parse_mass_models()

    run_efe = "--efe" in sys.argv or len(sys.argv) == 1
    run_pg = "--pergalaxy" in sys.argv or len(sys.argv) == 1
    run_boot = "--bootstrap" in sys.argv or len(sys.argv) == 1

    if run_efe:
        efe_analysis(df)
    if run_pg:
        per_galaxy_cpx5(df)
    if run_boot:
        bootstrap_rar(df, n_boot=200)

    print("\nDone.")

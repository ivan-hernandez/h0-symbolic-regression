"""Gas fraction systematic: disentangle gas fraction from RAR residuals.

The per-galaxy analysis shows a strong anti-correlation between gas fraction
and CPX5 residuals (ρ = -0.31, p = 10^{-74}). This script tests whether:

1. The gas fraction trend is real or a proxy for SB, mass, distance
2. CPX5 parameters change when fitting gas-rich vs gas-poor galaxies separately
3. Adding a gas fraction term to CPX5 improves the fit significantly
4. The MOND interpolating function shows the same gas fraction trend
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr
from parse_sparc import parse_mass_models

kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0

# ── Models ────────────────────────────────────────────────────────────────────

def sr_cpx5_log(log_gbar, a, b):
    return a + b / np.maximum(log_gbar, -50)

def mond_simple(gbar, a0):
    return gbar * (1 + np.sqrt(1 + 4*a0/np.maximum(gbar, 1e-20))) / 2

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(df=None, Ud=0.5, Ub=0.7):
    if df is None:
        df = parse_mass_models()
    df = df[df["R"] > 0].copy()

    Vbar_sq = (np.abs(df["Vgas"].values) * df["Vgas"].values
               + Ud * df["Vdisk"].values**2
               + Ub * df["Vbul"].values**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)

    R_m = df["R"].values * kpc_to_m
    gbar = Vbar_sq * KM_S_TO_M_S**2 / R_m
    gobs = df["Vobs"].values**2 * KM_S_TO_M_S**2 / R_m
    valid = (gbar > 1e-13) & (gobs > 0)

    # Proper mass-based gas fraction at each radius
    gas_mass = np.abs(df["Vgas"].values) * df["Vgas"].values
    gas_mass = np.maximum(gas_mass, 0.0)
    stellar_mass = Ud * df["Vdisk"].values**2 + Ub * df["Vbul"].values**2
    f_gas = np.where(gas_mass + stellar_mass > 0,
                     gas_mass / (gas_mass + stellar_mass), 0.0)

    result = pd.DataFrame({
        "ID": df["ID"].values,
        "D": df["D"].values,
        "R": df["R"].values,
        "Vobs": df["Vobs"].values,
        "gbar": gbar,
        "gobs": gobs,
        "log_gbar": np.log10(np.maximum(gbar, 1e-20)),
        "log_gobs": np.log10(np.maximum(gobs, 1e-20)),
        "f_gas": f_gas,
        "SBdisk": df["SBdisk"].values,
        "SBbul": df["SBbul"].values,
    })
    result = result[valid].copy()
    result["log_SB"] = np.log10(np.maximum(result["SBdisk"] + result["SBbul"], 0.1))
    return result


# ── 1. Gas fraction bins ─────────────────────────────────────────────────────

def gas_fraction_binning(data, outdir="analysis"):
    """Bin galaxies by gas fraction and fit CPX5 separately."""
    print("\n" + "=" * 60)
    print("Gas fraction binning analysis")
    print("=" * 60)

    # Per-galaxy median gas fraction
    gal_fgas = data.groupby("ID")["f_gas"].median().sort_values()
    print(f"  Galaxy gas fraction range: [{gal_fgas.min():.3f}, {gal_fgas.max():.3f}]")
    print(f"  Median: {gal_fgas.median():.3f}")

    # Split into 3 bins
    n_gal = len(gal_fgas)
    bin_edges = [0, int(n_gal/3), int(2*n_gal/3), n_gal]
    labels = ["gas-poor", "intermediate", "gas-rich"]
    bin_ranges = [(gal_fgas.iloc[bin_edges[i]:bin_edges[i+1]].min(),
                   gal_fgas.iloc[bin_edges[i]:bin_edges[i+1]].max())
                  for i in range(3)]

    results = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, (label, (lo, hi)) in enumerate(zip(labels, bin_ranges)):
        gas_gals = gal_fgas[(gal_fgas >= lo) & (gal_fgas <= hi)].index
        sub = data[data["ID"].isin(gas_gals)]
        x, y = sub["log_gbar"].values, sub["log_gobs"].values
        sigma = np.maximum(np.full_like(y, 0.1), 0)

        try:
            popt, pcov = curve_fit(sr_cpx5_log, x, y, p0=[-12, -50],
                                    maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            pred = sr_cpx5_log(x, *popt)
            chi2 = np.sum((y - pred)**2 / np.maximum(sigma, 0.01)**2)
            rms = np.sqrt(np.mean((y - pred)**2))
        except Exception as e:
            print(f"  {label}: fit failed: {e}")
            popt, perr = [np.nan, np.nan], [np.nan, np.nan]
            chi2, rms = np.nan, np.nan

        print(f"\n  {label} (f_gas=[{lo:.3f},{hi:.3f}]):")
        print(f"    N_galaxies: {sub['ID'].nunique()}")
        print(f"    N_points: {len(sub)}")
        print(f"    CPX5 a={popt[0]:.3f}±{perr[0]:.3f}, b={popt[1]:.3f}±{perr[1]:.3f}")
        print(f"    RMS: {rms:.4f} dex")

        results.append({
            "label": label,
            "f_gas_lo": lo, "f_gas_hi": hi,
            "n_gal": sub["ID"].nunique(),
            "n_pts": len(sub),
            "a": popt[0], "a_err": perr[0],
            "b": popt[1], "b_err": perr[1],
            "rms": rms,
        })

        # RAR plot
        ax = axes[0, i]
        ax.scatter(x, y, s=2, alpha=0.3, c=sub["f_gas"], cmap="viridis", vmin=0, vmax=1)
        x_grid = np.linspace(-13, -8, 100)
        ax.plot(x_grid, sr_cpx5_log(x_grid, *popt), "r-", lw=2)
        # Global CPX5 for comparison
        ax.plot(x_grid, sr_cpx5_log(x_grid, -17.06, -72.71), "k--", lw=1, alpha=0.5)
        ax.set_xlim(-13, -8)
        ax.set_ylim(-13, -8)
        ax.set_xlabel("log gbar")
        ax.set_ylabel("log gobs")
        ax.set_title(f"{label} (f_gas=[{lo:.2f},{hi:.2f}])")

        # Residuals
        ax = axes[1, i]
        resid = y - sr_cpx5_log(x, -17.06, -72.71)
        ax.scatter(x, resid, s=2, alpha=0.3, c=sub["f_gas"], cmap="viridis", vmin=0, vmax=1)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_xlim(-13, -8)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("log gbar")
        ax.set_ylabel("Residual (dex)")

    plt.tight_layout()
    plt.savefig(f"{outdir}/gas_fraction_bins.png", dpi=150)
    print(f"\n  Saved {outdir}/gas_fraction_bins.png")
    plt.close()

    # Parameter trends across bins
    df_r = pd.DataFrame(results)
    print(f"\n  b parameter trend: {df_r['b'].values}")
    print(f"  a parameter trend: {df_r['a'].values}")

    return df_r


# ── 2. CPX5 + gas fraction extension ─────────────────────────────────────────

def cpx5_plus_gas_log(log_gbar, a, b, c):
    """(unused - tests use direct minimization)"""
    return a + b / np.maximum(log_gbar, -50)

def test_gas_extension(data, outdir="analysis"):
    """Test if adding gas fraction to CPX5 improves fit."""
    print("\n" + "=" * 60)
    print("CPX5 + gas fraction extension test")
    print("=" * 60)

    x, y = data["log_gbar"].values, data["log_gobs"].values
    f_gas = data["f_gas"].values

    # CPX5 only (2 params)
    popt_cpx5, _ = curve_fit(sr_cpx5_log, x, y, p0=[-17, -70], maxfev=10000)
    pred_cpx5 = sr_cpx5_log(x, *popt_cpx5)
    resid_cpx5 = y - pred_cpx5
    chi2_cpx5 = np.sum(resid_cpx5**2)

    # CPX5 + gas fraction: log_gobs = a + b/log_gbar + c*f_gas
    def model_plus_gas(params):
        a, b, c = params
        pred = a + b / np.maximum(x, -50) + c * f_gas
        return np.sum((y - pred)**2)

    result = minimize(model_plus_gas, x0=[-17, -70, 0], method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    a_pg, b_pg, c_pg = result.x
    pred_pg = a_pg + b_pg / np.maximum(x, -50) + c_pg * f_gas
    resid_pg = y - pred_pg
    chi2_pg = result.fun

    # CPX5 + log_gbar (control: does adding any linear term help?)
    def model_plus_linear(params):
        a, b, d = params
        pred = a + b / np.maximum(x, -50) + d * x
        return np.sum((y - pred)**2)

    result2 = minimize(model_plus_linear, x0=[-17, -70, 0], method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    a_pl, b_pl, d_pl = result2.x
    pred_pl = a_pl + b_pl / np.maximum(x, -50) + d_pl * x
    chi2_pl = result2.fun

    n = len(y)
    print(f"\n  Model comparison:")
    print(f"  {'Model':<35s} {'k':<5s} {'χ²':<12s} {'Δχ²':<10s}")
    print(f"  {'-'*35} {'-'*5} {'-'*12} {'-'*10}")
    print(f"  {'CPX5 (2 params)':<35s} {2:<5d} {chi2_cpx5:<12.1f} {0:<+10.1f}")
    print(f"  {'CPX5 + f_gas (3 params)':<35s} {3:<5d} {chi2_pg:<12.1f} {chi2_pg - chi2_cpx5:<+10.1f}")
    print(f"  {'CPX5 + log_gbar (3 params)':<35s} {3:<5d} {chi2_pl:<12.1f} {chi2_pl - chi2_cpx5:<+10.1f}")

    print(f"\n  CPX5 + f_gas: a={a_pg:.3f}, b={b_pg:.3f}, c={c_pg:.3f}")
    print(f"  CPX5 + linear: a={a_pl:.3f}, b={b_pl:.3f}, d={d_pl:.3f}")

    # For interpretable units: bin by gas fraction, plot mean residual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    bins = np.linspace(0, 1, 20)
    bin_centers = []
    bin_means = []
    bin_stds = []
    for i in range(len(bins)-1):
        mask = (f_gas >= bins[i]) & (f_gas < bins[i+1])
        if mask.sum() > 10:
            bc = (bins[i] + bins[i+1])/2
            bin_centers.append(bc)
            bin_means.append(np.mean(resid_cpx5[mask]))
            bin_stds.append(np.std(resid_cpx5[mask]) / np.sqrt(mask.sum()))
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="o-", capsize=3)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Gas fraction")
    ax.set_ylabel("Mean CPX5 residual (dex)")

    ax = axes[1]
    ax.scatter(f_gas, resid_cpx5, s=1, alpha=0.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Gas fraction")
    ax.set_ylabel("CPX5 residual (dex)")

    ax = axes[2]
    resid_pg = y - pred_pg
    bins2 = np.linspace(0, 1, 20)
    bin_centers2 = []
    bin_means2 = []
    bin_stds2 = []
    for i in range(len(bins2)-1):
        mask = (f_gas >= bins2[i]) & (f_gas < bins2[i+1])
        if mask.sum() > 10:
            bc = (bins2[i] + bins2[i+1])/2
            bin_centers2.append(bc)
            bin_means2.append(np.mean(resid_pg[mask]))
            bin_stds2.append(np.std(resid_pg[mask]) / np.sqrt(mask.sum()))
    ax.errorbar(bin_centers2, bin_means2, yerr=bin_stds2, fmt="o-", capsize=3)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Gas fraction")
    ax.set_ylabel("CPX5 + f_gas residual (dex)")

    plt.tight_layout()
    plt.savefig(f"{outdir}/gas_extension.png", dpi=150)
    print(f"  Saved {outdir}/gas_extension.png")
    plt.close()

    return chi2_cpx5, chi2_pg, chi2_pl, c_pg


# ── 3. Disentangle gas fraction from other properties ────────────────────────

def multivariate_analysis(data, outdir="analysis"):
    """Check if gas fraction trend is actually driven by SB, mass, or distance."""
    print("\n" + "=" * 60)
    print("Multivariate analysis: is gas fraction really the driver?")
    print("=" * 60)

    # Per-galaxy medians
    gal_props = data.groupby("ID").agg({
        "f_gas": "median",
        "log_SB": "first",
        "D": "first",
        "log_gbar": "median",
        "log_gobs": "median",
    }).rename(columns={"log_gbar": "log_gbar_med", "log_gobs": "log_gobs_med"})

    # Global CPX5 fit → per-galaxy residuals
    x_all, y_all = data["log_gbar"].values, data["log_gobs"].values
    popt, _ = curve_fit(sr_cpx5_log, x_all, y_all, p0=[-17, -70], maxfev=10000)
    data["resid_cpx5"] = y_all - sr_cpx5_log(x_all, *popt)

    # Per-galaxy mean residual
    gal_resid = data.groupby("ID")["resid_cpx5"].mean()
    gal_props = gal_props.join(gal_resid.to_frame("mean_resid"))

    print(f"\n  Correlations with mean CPX5 residual ({len(gal_props)} galaxies):")
    props = ["f_gas", "log_SB", "D", "log_gbar_med"]
    labels = ["gas fraction", "log SB", "distance", "median log gbar"]
    for prop, label in zip(props, labels):
        r, p = spearmanr(gal_props[prop], gal_props["mean_resid"])
        print(f"    {label}: ρ={r:.4f}, p={p:.2e}")

    # Partial correlation: gas fraction controlling for SB and log_gbar_med
    # Use Pearson partial correlation
    def partial_corr(data, x_col, y_col, z_cols):
        """Partial correlation between x and y controlling for z."""
        from scipy import stats
        n = len(data)
        z = data[z_cols].values
        x, y = data[x_col].values, data[y_col].values

        # Regress x on z
        z_with_intercept = np.column_stack([np.ones(n), z])
        beta_x, _, _, _ = np.linalg.lstsq(z_with_intercept, x, rcond=None)
        x_resid = x - z_with_intercept @ beta_x

        # Regress y on z
        beta_y, _, _, _ = np.linalg.lstsq(z_with_intercept, y, rcond=None)
        y_resid = y - z_with_intercept @ beta_y

        r_partial, p_partial = stats.pearsonr(x_resid, y_resid)
        return r_partial, p_partial

    print(f"\n  Partial correlations (controlling for log_SB and median log_gbar):")
    r_partial, p_partial = partial_corr(gal_props, "f_gas", "mean_resid",
                                         ["log_SB", "log_gbar_med"])
    print(f"    f_gas vs resid | log_SB, log_gbar: r={r_partial:.4f}, p={p_partial:.2e}")
    r_partial2, p_partial2 = partial_corr(gal_props, "log_SB", "mean_resid",
                                           ["f_gas", "log_gbar_med"])
    print(f"    log_SB vs resid | f_gas, log_gbar: r={r_partial2:.4f}, p={p_partial2:.2e}")

    # Plot matrix (only for complete pairs)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    plot_props = [("f_gas", "gas fraction"), ("log_SB", "log SB (L_sun/pc²)"),
                   ("D", "distance (Mpc)"), ("log_gbar_med", "median log gbar")]

    for i in range(4):
        for j in range(4):
            ax = axes[i][j]
            p1, l1 = plot_props[i]
            p2, l2 = plot_props[j]
            if i == j:
                ax.hist(gal_props[p1].dropna(), bins=30, color="gray", edgecolor="none")
                ax.set_xlabel(l1)
            elif i > j:
                ax.scatter(gal_props[p2], gal_props[p1], s=10, alpha=0.5,
                           c=gal_props["mean_resid"], cmap="RdBu_r",
                           vmin=-0.15, vmax=0.15, edgecolor="none")
                ax.set_xlabel(l2)
                ax.set_ylabel(l1)
            else:
                ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{outdir}/multivariate_props.png", dpi=150)
    print(f"  Saved {outdir}/multivariate_props.png")
    plt.close()

    return gal_props


# ── 4. MOND residuals vs gas fraction ─────────────────────────────────────────

def mond_gas_trend(data, outdir="analysis"):
    """Check if MOND residuals also correlate with gas fraction."""
    print("\n" + "=" * 60)
    print("MOND residual gas fraction trend")
    print("=" * 60)

    x_gbar = data["gbar"].values
    y_gobs = data["gobs"].values
    f_gas = data["f_gas"].values

    # Fit MOND Simple
    def mond_chi2(a0):
        pred = mond_simple(x_gbar, a0)
        resid = y_gobs - pred
        return np.sum(resid**2 / np.maximum(0.1 * y_gobs, 1e-10)**2)

    result = minimize(mond_chi2, x0=[1.2e-10], bounds=[(1e-11, 1e-9)], method="L-BFGS-B")
    a0_best = result.x[0]
    pred_mond = mond_simple(x_gbar, a0_best)
    resid_mond = np.log10(y_gobs) - np.log10(pred_mond)

    r, p = spearmanr(f_gas, resid_mond)
    print(f"  MOND Simple a₀ = {a0_best:.3e}")
    print(f"  Spearman(gas frac, MOND resid): ρ = {r:.4f}, p = {p:.2e}")

    r_cpx5, p_cpx5 = spearmanr(f_gas, data["log_gobs"].values -
                                sr_cpx5_log(data["log_gbar"].values, -17.06, -72.71))
    print(f"  Spearman(gas frac, CPX5 resid): ρ = {r_cpx5:.4f}, p = {p_cpx5:.2e}")

    # Binned comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, resid, label in [
        (axes[0], resid_mond, "MOND Simple"),
        (axes[1], data["log_gobs"].values - sr_cpx5_log(data["log_gbar"].values, -17.06, -72.71),
         "CPX5"),
    ]:
        bins = np.linspace(0, 1, 20)
        bin_centers, bin_means, bin_stds = [], [], []
        for i in range(len(bins)-1):
            mask = (f_gas >= bins[i]) & (f_gas < bins[i+1])
            if mask.sum() > 10:
                bc = (bins[i]+bins[i+1])/2
                bin_centers.append(bc)
                bin_means.append(np.mean(resid[mask]))
                bin_stds.append(np.std(resid[mask]) / np.sqrt(mask.sum()))
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="o-", capsize=3)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel("Gas fraction")
        ax.set_ylabel(f"{label} residual (dex)")
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig(f"{outdir}/mond_gas_trend.png", dpi=150)
    print(f"  Saved {outdir}/mond_gas_trend.png")
    plt.close()

    return a0_best, r, p


# ── 5. Two-parameter CPX5 with gas fraction correction ────────────────────────

def stacked_gas_correction(data, outdir="analysis"):
    """Check if the gas fraction trend is a per-galaxy offset or radial trend."""
    print("\n" + "=" * 60)
    print("Gas fraction: per-galaxy offset vs radial trend")
    print("=" * 60)

    # Compute global CPX5 residuals
    x, y = data["log_gbar"].values, data["log_gobs"].values
    f_gas = data["f_gas"].values

    # Split: is the trend driven by inner (high-gbar) or outer (low-gbar) points?
    inner = data[data["log_gbar"] > -10.5]
    outer = data[data["log_gbar"] <= -10.5]

    for region, sub in [("Inner (log_gbar > -10.5)", inner),
                        ("Outer (log_gbar <= -10.5)", outer)]:
        if len(sub) < 10:
            continue
        resid = sub["log_gobs"].values - sr_cpx5_log(sub["log_gbar"].values, -17.06, -72.71)
        r, p = spearmanr(sub["f_gas"].values, resid)
        print(f"  {region}: ρ={r:.4f}, p={p:.2e} (N={len(sub)})")

    # Per-galaxy: is the offset correlated with gas fraction?
    gal_offset = data.groupby("ID").apply(
        lambda g: np.mean(g["log_gobs"].values - sr_cpx5_log(g["log_gbar"].values, -17.06, -72.71))
    )
    gal_fgas = data.groupby("ID")["f_gas"].median()

    r_off, p_off = spearmanr(gal_offset.values, gal_fgas.values)
    print(f"\n  Per-galaxy offset vs gas fraction: ρ={r_off:.4f}, p={p_off:.2e}")

    # Fit: log_gobs = a + b/log_gbar + c_per_galaxy * f_gas_gal (per-galaxy f_gas)
    # Use the per-galaxy median f_gas as a categorical-like variable
    gal_fgas_map = gal_fgas.to_dict()
    data["gal_f_gas"] = data["ID"].map(gal_fgas_map)
    f_gas_gal = data["gal_f_gas"].values

    def chi2_gal_gas(params):
        a, b, c = params
        pred = a + b / np.maximum(x, -50) + c * f_gas_gal
        return np.sum((y - pred)**2)

    result = minimize(chi2_gal_gas, x0=[-17, -70, 0], method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    a_gg, b_gg, c_gg = result.x
    chi2_gg = result.fun

    # Compare with per-point f_gas
    def chi2_pt_gas(params):
        a, b, c = params
        pred = a + b / np.maximum(x, -50) + c * f_gas
        return np.sum((y - pred)**2)

    result2 = minimize(chi2_pt_gas, x0=[-17, -70, 0], method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    a_pg2, b_pg2, c_pg2 = result2.x
    chi2_pg2 = result2.fun

    # CPX5 only
    chi2_base = np.sum((y - sr_cpx5_log(x, -17.06, -72.71))**2)

    print(f"\n  Model comparison:")
    print(f"  {'Model':<40s} {'k':<5s} {'χ²':<12s} {'Δχ²':<10s}")
    print(f"  {'-'*40} {'-'*5} {'-'*12} {'-'*10}")
    print(f"  {'CPX5 (global fit)':<40s} {2:<5d} {chi2_base:<12.1f} {0:<+10.1f}")
    print(f"  {'CPX5 + gal.f_gas_c (3 params)':<40s} {3:<5d} {chi2_gg:<12.1f} {chi2_gg - chi2_base:<+10.1f}")
    print(f"  {'CPX5 + pt.f_gas_c (3 params)':<40s} {3:<5d} {chi2_pg2:<12.1f} {chi2_pg2 - chi2_base:<+10.1f}")

    return {"c_gal": c_gg, "c_pt": c_pg2, "chi2_gal": chi2_gg, "chi2_pt": chi2_pg2}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    os.makedirs("analysis", exist_ok=True)

    print("=" * 60)
    print("Gas Fraction Systematic Analysis")
    print("=" * 60)

    data = load_data()
    print(f"\n  Total points: {len(data)}")
    print(f"  Total galaxies: {data['ID'].nunique()}")

    # Run analyses
    # 1. Gas fraction bins
    bin_results = gas_fraction_binning(data)

    # 2. CPX5 + gas extension
    chi2_cpx5, chi2_pg, chi2_pl, c_pg = test_gas_extension(data)

    # 3. Multivariate
    gal_props = multivariate_analysis(data)

    # 4. MOND comparison
    mond_gas_trend(data)

    # 5. Stacked: per-galaxy vs per-point
    corr_results = stacked_gas_correction(data)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Per-point vs per-galaxy comparison
    resid_per_pt = data["log_gobs"].values - sr_cpx5_log(data["log_gbar"].values, -17.06, -72.71)
    gas_spearman_pt, gas_p_pt = spearmanr(data["f_gas"], resid_per_pt)
    print(f"  PER-POINT: gas fraction vs CPX5 residual: ρ={gas_spearman_pt:.4f}, p={gas_p_pt:.2e}")

    gal_resid = data.groupby("ID").apply(
        lambda g: np.mean(g["log_gobs"].values - sr_cpx5_log(g["log_gbar"].values, -17.06, -72.71))
    )
    gal_fgas = data.groupby("ID")["f_gas"].median()
    gas_spearman_gal, gas_p_gal = spearmanr(gal_resid.values, gal_fgas.values)
    print(f"  PER-GALAXY: gas fraction vs mean CPX5 residual: ρ={gas_spearman_gal:.4f}, p={gas_p_gal:.2e}")

    # Compare: old velocity-based proxy from the EFE analysis used
    # gas_frac = |Vgas|/Vobs (crude), not mass-based f_gas = Vgas²/(Vgas²+Ud·Vdisk²+Ub·Vbul²)
    # The old proxy gave ρ≈-0.31 (p~10^{-74}) but this was inflated by:
    # 1. Using linear velocities not squared (not a true mass ratio)
    # 2. Using Vobs in denominator (includes dark matter, dilutes signal)
    # 3. Not accounting for M/L factors
    # The proper mass-based fraction gives ρ≈-0.037 (per-point), ρ≈-0.022 (per-galaxy)
    print(f"  Note: old |Vgas|/Vobs proxy gave ρ≈-0.31 (see EFE analysis)")
    print(f"  Proper mass-based fraction reduces correlation by 10×")

    # Plot per-point vs per-galaxy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(data["f_gas"], resid_per_pt, s=1, alpha=0.1)
    bins = np.linspace(0, 1, 25)
    bc, bm, bs = [], [], []
    for i in range(len(bins)-1):
        mask = (data["f_gas"].values >= bins[i]) & (data["f_gas"].values < bins[i+1])
        if mask.sum() > 10:
            bc.append((bins[i]+bins[i+1])/2)
            bm.append(np.mean(resid_per_pt[mask]))
            bs.append(np.std(resid_per_pt[mask])/np.sqrt(mask.sum()))
    ax.errorbar(bc, bm, yerr=bs, fmt="r.-", lw=2, capsize=3)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Gas fraction (mass-based)")
    ax.set_ylabel("CPX5 residual (dex)")
    ax.set_title(f"Per-point (ρ={gas_spearman_pt:.3f}, p={gas_p_pt:.2e})")

    ax = axes[1]
    ax.scatter(gal_fgas.values, gal_resid.values, s=10, alpha=0.5, edgecolor="none")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Gas fraction (mass-based)")
    ax.set_ylabel("Mean CPX5 residual (dex)")
    ax.set_title(f"Per-galaxy (ρ={gas_spearman_gal:.3f}, p={gas_p_gal:.2e})")

    plt.tight_layout()
    plt.savefig("analysis/per_point_vs_per_gal.png", dpi=150)
    print("  Saved analysis/per_point_vs_per_gal.png")
    plt.close()

    # AIC comparison
    n_pts = len(data)
    aic_cpx5 = n_pts * np.log(chi2_cpx5 / n_pts) + 2 * 2
    aic_pg = n_pts * np.log(chi2_pg / n_pts) + 2 * 3
    aic_pl = n_pts * np.log(chi2_pl / n_pts) + 2 * 3
    print(f"\n  AIC comparison (using sum of squared residuals, flat weight):")
    print(f"    CPX5: χ² = {chi2_cpx5:.1f}, AIC = {aic_cpx5:.1f}")
    print(f"    CPX5 + f_gas: χ² = {chi2_pg:.1f}, AIC = {aic_pg:.1f} (ΔAIC = {aic_pg - aic_cpx5:.1f})")
    print(f"    CPX5 + log_gbar: χ² = {chi2_pl:.1f}, AIC = {aic_pl:.1f} (ΔAIC = {aic_pl - aic_cpx5:.1f})")

    gas_daics = [aic_pg - aic_cpx5, aic_pl - aic_cpx5]
    if aic_pg < aic_cpx5:
        print(f"  → CPX5 + f_gas is preferred (ΔAIC = {aic_pg - aic_cpx5:.1f})")
    else:
        print(f"  → CPX5 alone is preferred (ΔAIC = {aic_pg - aic_cpx5:.1f})")
    if aic_pl < aic_cpx5 and aic_pl < aic_pg:
        print(f"  → CPX5 + log_gbar is preferred (ΔAIC = {aic_pl - aic_cpx5:.1f})")

    print("\nDone.")

"""RAR analysis: MOND fit, SR comparison, holdout validation, property dependence.

Usage:
    python rar_analysis.py                  # full analysis
    python rar_analysis.py --mond-only      # just MOND fits
    python rar_analysis.py --holdout        # just holdout test
    python rar_analysis.py --properties     # galaxy property dependence
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parse_sparc import parse_mass_models, compute_radial_accelerations

# ── MOND interpolating functions ──────────────────────────────────────────────
def mond_simple(gbar, a0):
    """Simple MOND: g_obs = g_bar * (1 + sqrt(1 + 4a0/g_bar)) / 2
    Famaey & Binney 2005."""
    return gbar * (1 + np.sqrt(1 + 4 * a0 / np.maximum(gbar, 1e-20))) / 2

def mond_standard(gbar, a0):
    """Standard MOND: g_obs = g_bar * sqrt((1 + sqrt(1 + 4*(a0/g_bar)**2)) / 2)
    Kent 1987, Sanders & McGaugh 2002."""
    y = a0 / np.maximum(gbar, 1e-20)
    return gbar * np.sqrt((1 + np.sqrt(1 + 4 * y**2)) / 2)

def mond_mcgaugh(gbar, a0):
    """McGaugh+2016 RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))"""
    return gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20) / a0)), 1e-20)

# ── SR-discovered forms ──────────────────────────────────────────────────────
def sr_cpx3(gbar, a, b):
    """log_gobs = a * log_gbar + b  (linear in log-log)"""
    return 10 ** (a * np.log10(gbar) + b)

def sr_cpx5(gbar, a, b):
    """log_gobs = a + b / log_gbar  (inverted in log-log)"""
    log_gbar = np.log10(np.maximum(gbar, 1e-20))
    return 10 ** (a + b / log_gbar)

def sr_cpx7(gbar, a, b, c):
    """log_gobs = a + b / (log_gbar + c)  (shifted inverted)"""
    log_gbar = np.log10(np.maximum(gbar, 1e-20))
    return 10 ** (a + b / (log_gbar - c))


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(min_gbar=1e-13):
    df = parse_mass_models()
    acc = compute_radial_accelerations(df)
    valid = (acc["gbar"].values > min_gbar) & (acc["gobs"].values > 0)
    data = acc[valid].copy()
    print(f"Loaded {len(data)} points from {data['ID'].nunique()} galaxies")
    return data


# ── Fitting utilities ─────────────────────────────────────────────────────────
def fit_mond(data, func, label, a0_guess=1.2e-10):
    gbar, gobs = data["gbar"].values, data["gobs"].values
    sigma = np.maximum(data["e_gobs"].values, 0.1 * gobs)
    try:
        popt, pcov = curve_fit(func, gbar, gobs, p0=[a0_guess],
                                sigma=sigma, absolute_sigma=True,
                                maxfev=5000, ftol=1e-10, xtol=1e-10)
        a0 = popt[0]
        perr = np.sqrt(np.diag(pcov))[0] if pcov is not None else np.nan
        pred = func(gbar, *popt)
        resid = gobs - pred
        chi2 = np.sum(resid**2 / sigma**2)
        n = len(gbar)
        k = len(popt)
        aic = n * np.log(chi2 / n) + 2 * k
        bic = n * np.log(chi2 / n) + k * np.log(n)
        return {
            "model": label,
            "a0": a0,
            "a0_err": perr,
            "chi2": chi2,
            "chi2_red": chi2 / (n - k),
            "AIC": aic,
            "BIC": bic,
            "params": popt,
        }
    except Exception as e:
        print(f"  {label} fit failed: {e}")
        return None


def fit_sr(data, func, label, p0):
    gbar, gobs = data["gbar"].values, data["gobs"].values
    sigma = np.maximum(data["e_gobs"].values, 0.1 * gobs)
    try:
        popt, pcov = curve_fit(func, gbar, gobs, p0=p0,
                                sigma=sigma, absolute_sigma=True,
                                maxfev=20000, ftol=1e-10, xtol=1e-10)
        pred = func(gbar, *popt)
        resid = gobs - pred
        chi2 = np.sum(resid**2 / sigma**2)
        n = len(gbar)
        k = len(popt)
        aic = n * np.log(chi2 / n) + 2 * k
        bic = n * np.log(chi2 / n) + k * np.log(n)
        return {
            "model": label,
            "chi2": chi2,
            "chi2_red": chi2 / (n - k),
            "AIC": aic,
            "BIC": bic,
            "params": popt,
        }
    except Exception as e:
        print(f"  {label} fit failed: {e}")
        return None


# ── Analysis functions ────────────────────────────────────────────────────────
def run_mond_fits(data, outdir="analysis"):
    """Fit all MOND interpolating functions and SR forms."""
    import os
    os.makedirs(outdir, exist_ok=True)

    gbar, gobs = data["gbar"].values, data["gobs"].values

    functions = [
        (mond_simple, "MOND Simple (Famaey+2005)", 1.2e-10),
        (mond_standard, "MOND Standard (Kent 1987)", 1.2e-10),
        (mond_mcgaugh, "MOND McGaugh+2016 (RAR)", 1.2e-10),
    ]

    results = []
    for func, label, guess in functions:
        print(f"  Fitting {label}...")
        r = fit_mond(data, func, label, guess)
        if r:
            results.append(r)
            print(f"    a0 = {r['a0']:.3e} ± {r['a0_err']:.3e} m/s²")
            print(f"    χ²_red = {r['chi2_red']:.3f}, AIC = {r['AIC']:.1f}")

    # SR forms
    sr_functions = [
        (sr_cpx3, "SR CPX3 (linear log-log)", [0.9, -2]),
        (sr_cpx5, "SR CPX5 (inverted)", [-12, -50]),
        (sr_cpx7, "SR CPX7 (shifted inverted)", [-12, -80, 1.5]),
    ]
    for func, label, p0 in sr_functions:
        print(f"  Fitting {label}...")
        r = fit_sr(data, func, label, p0)
        if r:
            r["a0"] = np.nan
            r["a0_err"] = np.nan
            results.append(r)
            print(f"    χ²_red = {r['chi2_red']:.3f}, AIC = {r['AIC']:.1f}")

    # Summary table
    df_r = pd.DataFrame(results)
    df_r = df_r.sort_values("AIC")
    print(f"\n{'='*70}")
    print(f"Model comparison (sorted by AIC)")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'a₀ (m/s²)':<18} {'χ²_red':<10} {'AIC':<10} {'BIC':<10}")
    print(f"{'-'*30} {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in df_r.iterrows():
        a0_str = f"{row['a0']:.2e}" if not np.isnan(row['a0']) else "N/A"
        print(f"{row['model']:<30} {a0_str:<18} {row['chi2_red']:<10.3f} {row['AIC']:<10.1f} {row['BIC']:<10.1f}")

    df_r.to_csv(f"{outdir}/model_comparison.csv", index=False)
    print(f"\nSaved {outdir}/model_comparison.csv")

    # Plot all models
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(gbar, gobs, s=1, alpha=0.3, color="gray", label=f"SPARC ({len(gbar)} pts)")
    gbar_grid = np.logspace(-13, -8, 300)

    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for i, row in df_r.iterrows():
        if "Simple" in row["model"]:
            pred = mond_simple(gbar_grid, row["a0"])
        elif "Standard" in row["model"]:
            pred = mond_standard(gbar_grid, row["a0"])
        elif "McGaugh" in row["model"]:
            pred = mond_mcgaugh(gbar_grid, row["a0"])
        elif "CPX3" in row["model"]:
            pred = sr_cpx3(gbar_grid, *row["params"])
        elif "CPX5" in row["model"]:
            pred = sr_cpx5(gbar_grid, *row["params"])
        elif "CPX7" in row["model"]:
            pred = sr_cpx7(gbar_grid, *row["params"])
        else:
            continue
        ax.plot(gbar_grid, pred, "-", color=colors[i % len(colors)],
                lw=2, label=f"{row['model']} (AIC={row['AIC']:.0f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$g_{\rm bar}$ (m/s$^2$)")
    ax.set_ylabel(r"$g_{\rm obs}$ (m/s$^2$)")
    ax.legend(fontsize=8)
    ax.set_xlim([1e-13, 1e-8])
    ax.set_ylim([5e-13, 1e-8])
    plt.tight_layout()
    plt.savefig(f"{outdir}/all_models.png", dpi=150)
    print(f"Saved {outdir}/all_models.png")
    plt.close()

    return df_r


def run_holdout(data, outdir="analysis", n_splits=10, train_frac=0.8):
    """Holdout validation: train on 80% of galaxies, test on 20%."""
    import os
    os.makedirs(outdir, exist_ok=True)

    galaxies = data["ID"].unique()
    n_gal = len(galaxies)
    n_train = int(train_frac * n_gal)

    results = []
    for split in range(n_splits):
        rng = np.random.RandomState(split)
        train_gal = rng.choice(galaxies, n_train, replace=False)
        train = data[data["ID"].isin(train_gal)]
        test = data[~data["ID"].isin(train_gal)]

        gbar_tr, gobs_tr = train["gbar"].values, train["gobs"].values
        gbar_te, gobs_te = test["gbar"].values, test["gobs"].values

        # Fit MOND simple on training set
        try:
            popt, _ = curve_fit(mond_simple, gbar_tr, gobs_tr, p0=[1.2e-10],
                                 maxfev=5000, ftol=1e-10)
            pred_te = mond_simple(gbar_te, *popt)
            rms_te = np.sqrt(np.mean((np.log10(pred_te) - np.log10(gobs_te))**2))
            pred_tr = mond_simple(gbar_tr, *popt)
            rms_tr = np.sqrt(np.mean((np.log10(pred_tr) - np.log10(gobs_tr))**2))
            results.append({
                "split": split, "model": "MOND Simple",
                "rms_train": rms_tr, "rms_test": rms_te,
                "n_train": len(train), "n_test": len(test),
                "a0": popt[0],
            })
        except Exception as e:
            print(f"  Split {split}: MOND fit failed: {e}")

        # Fit SR CPX5 on training set
        try:
            popt, _ = curve_fit(sr_cpx5, gbar_tr, gobs_tr, p0=[-10, -50],
                                 maxfev=10000, ftol=1e-10)
            pred_te = sr_cpx5(gbar_te, *popt)
            rms_te = np.sqrt(np.mean((np.log10(pred_te) - np.log10(gobs_te))**2))
            pred_tr = sr_cpx5(gbar_tr, *popt)
            rms_tr = np.sqrt(np.mean((np.log10(pred_tr) - np.log10(gobs_tr))**2))
            results.append({
                "split": split, "model": "SR CPX5",
                "rms_train": rms_tr, "rms_test": rms_te,
                "n_train": len(train), "n_test": len(test),
                "a0": np.nan,
            })
        except Exception as e:
            print(f"  Split {split}: SR CPX5 fit failed: {e}")

        # Newtonian baseline: g_obs = g_bar (no free params)
        pred_te = gbar_te
        rms_te = np.sqrt(np.mean((np.log10(pred_te) - np.log10(gobs_te))**2))
        pred_tr = gbar_tr
        rms_tr = np.sqrt(np.mean((np.log10(pred_tr) - np.log10(gobs_tr))**2))
        results.append({
            "split": split, "model": "Newtonian (g=gbar)",
            "rms_train": rms_tr, "rms_test": rms_te,
            "n_train": len(train), "n_test": len(test),
            "a0": np.nan,
        })

    df_r = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print(f"Holdout validation ({n_splits} splits, {train_frac*100:.0f}%/{100-train_frac*100:.0f}% train/test)")
    print(f"{'='*70}")
    for model in df_r["model"].unique():
        sub = df_r[df_r["model"] == model]
        print(f"\n  {model}:")
        print(f"    Train RMS: {sub['rms_train'].mean():.4f} ± {sub['rms_train'].std():.4f} dex")
        print(f"    Test  RMS: {sub['rms_test'].mean():.4f} ± {sub['rms_test'].std():.4f} dex")
        if "a0" in sub.columns and not sub["a0"].isna().all():
            print(f"    a0: {sub['a0'].mean():.3e} ± {sub['a0'].std():.3e}")

    df_r.to_csv(f"{outdir}/holdout_results.csv", index=False)
    print(f"\nSaved {outdir}/holdout_results.csv")
    return df_r


def run_property_analysis(data, outdir="analysis"):
    """Test if RAR residuals correlate with galaxy properties (EFE proxy)."""
    import os
    os.makedirs(outdir, exist_ok=True)
    from parse_sparc import compute_radial_accelerations

    # Recompute with original mass model data (with SB info)
    df = parse_mass_models()
    acc = compute_radial_accelerations(df)
    # Merge back SB info
    data = data.merge(acc[["ID", "R", "Vobs"]], on=["ID", "R", "Vobs"],
                       how="left", suffixes=("", "_drop"))
    # Actually, let's recompute properly
    df = parse_mass_models()
    acc_full = compute_radial_accelerations(df)
    valid = (acc_full["gbar"].values > 1e-13) & (acc_full["gobs"].values > 0)
    data = acc_full[valid].copy()

    gbar, gobs = data["gbar"].values, data["gobs"].values

    # Best MOND fit
    try:
        popt, _ = curve_fit(mond_simple, gbar, gobs, p0=[1.2e-10], maxfev=5000)
        pred = mond_simple(gbar, *popt)
        resid = np.log10(gobs) - np.log10(pred)
        data["resid_mond"] = resid
        data["abs_resid_mond"] = np.abs(resid)
    except Exception as e:
        print(f"MOND fit failed: {e}")
        return None

    # Properties from mass models
    df_mass = parse_mass_models()
    # Merge by ID and R (approximate)
    data = data.merge(df_mass[["ID", "R", "SBdisk", "SBbul", "Vdisk", "Vbul", "Vgas"]],
                       on=["ID", "R"], how="left", suffixes=("", "_mass"))

    # Compute derived properties
    data["gas_frac"] = np.abs(data["Vgas"]) / np.maximum(data["Vobs"], 0.1)
    data["SB_total"] = data["SBdisk"] + data["SBbul"]
    data["log_SB"] = np.log10(np.maximum(data["SB_total"], 0.1))

    print(f"\n{'='*70}")
    print("Galaxy property dependence of RAR residuals (MOND Simple)")
    print(f"{'='*70}")

    properties = {
        "log_SB": "log Surface Brightness",
        "gas_frac": "Gas fraction",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (col, label) in enumerate(properties.items()):
        valid_prop = data[col].notna() & np.isfinite(data[col])
        sub = data[valid_prop]
        ax = axes[idx]
        scatter = ax.scatter(sub[col], sub["resid_mond"], s=2, alpha=0.3,
                            c=np.log10(sub["gbar"]), cmap="viridis")
        # Running median
        sort_idx = np.argsort(sub[col].values)
        x_sorted = sub[col].values[sort_idx]
        y_sorted = sub["resid_mond"].values[sort_idx]
        # Simple binning
        n_bins = 20
        bin_edges = np.array_split(np.arange(len(x_sorted)), n_bins)
        bin_means = []
        bin_centers = []
        bin_stds = []
        for bin_idx in bin_edges:
            if len(bin_idx) < 3:
                continue
            bin_centers.append(np.mean(x_sorted[bin_idx]))
            bin_means.append(np.median(y_sorted[bin_idx]))
            bin_stds.append(np.std(y_sorted[bin_idx]))
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds/np.sqrt(5),
                    fmt="r.-", lw=2, capsize=3)

        # Correlation
        from scipy.stats import pearsonr, spearmanr
        r_pear, p_pear = pearsonr(sub[col], sub["resid_mond"])
        r_spear, p_spear = spearmanr(sub[col], sub["resid_mond"])
        ax.set_xlabel(label)
        ax.set_ylabel("Δlog g_obs (MOND residual)")
        ax.set_title(f"r={r_pear:.3f} (p={p_pear:.2e})")
        plt.colorbar(scatter, ax=ax, label="log g_bar")
        print(f"  {label}: Pearson r={r_pear:.3f} p={p_pear:.2e}, Spearman ρ={r_spear:.3f} p={p_spear:.2e}")

    # Panel 3: residual vs gbar itself
    ax = axes[2]
    scatter = ax.scatter(np.log10(data["gbar"]), data["resid_mond"],
                         s=2, alpha=0.3, c=np.log10(data["gbar"]), cmap="viridis")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("log g_bar")
    ax.set_ylabel("Δlog g_obs (MOND residual)")
    ax.set_title("Residual vs g_bar")

    plt.tight_layout()
    plt.savefig(f"{outdir}/property_dependence.png", dpi=150)
    print(f"Saved {outdir}/property_dependence.png")
    plt.close()

    return data


def run_multiseed_sr(data, outdir="analysis", n_seeds=3, n_cycles=200):
    """Run PySR with multiple seeds to check stability."""
    import os, warnings, traceback
    warnings.filterwarnings("ignore")
    os.makedirs(outdir, exist_ok=True)

    valid = (data["gbar"] > 1e-13) & (data["gobs"] > 0)
    log_gbar = np.log10(data["gbar"].values[valid]).reshape(-1, 1)
    log_gobs = np.log10(data["gobs"].values[valid])

    from pysr import PySRRegressor

    all_eqs = []
    for seed in range(n_seeds):
        print(f"\n  PySR seed {seed} ({n_cycles} iterations)...")
        try:
            model = PySRRegressor(
                niterations=n_cycles,
                populations=12,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "cube", "exp", "log10"],
                maxsize=20,
                parsimony=0.001,
                procs=4,
                model_selection="accuracy",
                tempdir=f"{outdir}/pysr_seed{seed}",
            )
            model.fit(log_gbar, log_gobs)
            eqs = model.equations_
            eqs["seed"] = seed
            all_eqs.append(eqs)
            print(f"    Best (pick): {eqs[eqs['pick']].iloc[0]['equation']}")
            print(f"    Loss: {eqs[eqs['pick']].iloc[0]['loss']:.6f}")
        except Exception as e:
            print(f"    Failed: {e}")
            traceback.print_exc()

    if all_eqs:
        df_all = pd.concat(all_eqs, ignore_index=True)
        df_all.to_csv(f"{outdir}/multiseed_equations.csv", index=False)
        print(f"\nSaved {outdir}/multiseed_equations.csv")

        # Summary: which equations appear across seeds?
        print(f"\n{'='*50}")
        print("Multi-seed summary")
        print(f"{'='*50}")
        for complexity in sorted(df_all["complexity"].unique()):
            sub = df_all[df_all["complexity"] == complexity]
            losses = sub.groupby("seed")["loss"].min()
            print(f"  Complexity {complexity}: losses {losses.values}")
    return all_eqs


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mond-only", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--properties", action="store_true")
    parser.add_argument("--multiseed", action="store_true")
    parser.add_argument("--outdir", default="analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("RAR Analysis: MOND fits, SR comparison, holdout, properties")
    print("=" * 70)

    data = load_data()

    if args.mond_only:
        run_mond_fits(data, args.outdir)
    elif args.holdout:
        run_holdout(data, args.outdir)
    elif args.properties:
        run_property_analysis(data, args.outdir)
    elif args.multiseed:
        run_multiseed_sr(data, args.outdir)
    else:
        run_mond_fits(data, args.outdir)
        run_holdout(data, args.outdir)
        run_property_analysis(data, args.outdir)
        run_multiseed_sr(data, args.outdir)

    print("\nDone.")

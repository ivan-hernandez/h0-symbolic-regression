"""Phase 3: Hook permutation null test.

Fire-2 (Ardizzone+2023) predicts non-monotonic RAR tracks ("hooks") from
cored DM halos. We found 68% of SPARC galaxies show some hooks, but is this
more than expected from noise in poorly-sampled rotation curves?

This script generates null distributions by:
1. Fitting a smooth monotonic model (CPX5) to each galaxy's RAR track
2. Generating mock tracks with the same x-coordinates and scatter as the data
3. Applying the same hook detection algorithm
4. Repeating N_perm times per galaxy
5. Measuring how many galaxies exceed the null expectation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parse_sparc import parse_mass_models

OUTDIR = "analysis/phase3"
N_PERM = 500  # permutations per galaxy
RNG = np.random.RandomState(42)

def cpx5_log(x, a, b):
    return a + b / np.maximum(x, -50)


def load_data():
    """Load SPARC data with RAR coordinates."""
    df = parse_mass_models()
    kpc_to_m = 3.0857e19
    KM_S_TO_M_S = 1000.0
    df = df[df["R"] > 0].copy()
    Vbar_sq = (np.abs(df["Vgas"].values) * df["Vgas"].values
               + 0.5 * df["Vdisk"].values**2
               + 0.7 * df["Vbul"].values**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"].values * kpc_to_m
    gbar = Vbar_sq * KM_S_TO_M_S**2 / R_m
    gobs = df["Vobs"].values**2 * KM_S_TO_M_S**2 / R_m
    valid = (gbar > 1e-13) & (gobs > 0)
    return df[valid].copy(), gbar[valid], gobs[valid]


def detect_hooks(x, y):
    """Count hook points in a track. Same algorithm as hook_search.py."""
    if len(x) < 5:
        return 0.0, 0, 0, 0
    dx = np.diff(x)
    dy = np.diff(y)
    hook_points = 0
    sign_changes = 0
    for i in range(1, len(x)):
        if dx[i-1] < 0 and dy[i-1] > 0:
            hook_points += 1
    y_smooth = np.convolve(y, np.ones(3)/3, mode="valid")
    dy_smooth = np.diff(y_smooth)
    for i in range(1, len(dy_smooth)):
        if dy_smooth[i] * dy_smooth[i-1] < 0:
            sign_changes += 1
    hook_frac = hook_points / max(len(x) - 1, 1)
    return hook_frac, hook_points, sign_changes, len(x)


def run_null_test(outdir=OUTDIR):
    """Run permutation null test for hook features."""
    import os
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Phase 3: Hook Permutation Null Test")
    print(f"  Generating {N_PERM} null realizations per galaxy")
    print("=" * 60)

    df, gbar, gobs = load_data()
    df["log_gbar"] = np.log10(gbar)
    df["log_gobs"] = np.log10(gobs)

    print(f"\n  Total galaxies: {df['ID'].nunique()}")
    print(f"  Total points: {len(df)}")

    # Per-galaxy analysis
    galaxies = sorted(df["ID"].unique())
    n_gal = len(galaxies)
    n_with_hooks_observed = 0
    n_significant_hooks_observed = 0
    observed_hook_fractions = []
    null_hook_fractions_all = []
    results = []

    print(f"\n  Processing {n_gal} galaxies...")

    for g_idx, gal in enumerate(galaxies):
        sub = df[df["ID"] == gal].sort_values("R")
        x = sub["log_gbar"].values
        y = sub["log_gobs"].values
        n_pts = len(x)

        if n_pts < 5:
            continue

        # Observed hook fraction
        obs_hf, obs_hp, obs_sc, _ = detect_hooks(x, y)
        observed_hook_fractions.append(obs_hf)
        if obs_hf > 0:
            n_with_hooks_observed += 1
        if obs_hf > 0.1:
            n_significant_hooks_observed += 1

        # Fit CPX5 to get smooth trend + scatter
        try:
            popt, _ = curve_fit(cpx5_log, x, y, p0=[-12, -50], maxfev=5000)
            y_pred = cpx5_log(x, *popt)
            resid = y - y_pred
            rms = np.sqrt(np.mean(resid**2))
        except Exception:
            popt, rms = None, np.std(y) * 0.1
            y_pred = np.mean(y)

        # Upper bound check: if CPX5 fit fails, use running mean
        if rms < 0.001 or rms > 2.0 or np.isnan(rms):
            rms = np.mean(np.abs(np.diff(y))) * 1.5
            y_pred = np.convolve(y, np.ones(min(3, n_pts-1))/min(3, n_pts-1), mode="same")
            if len(y_pred) != len(x):
                y_pred = np.interp(np.linspace(0, 1, len(x)),
                                   np.linspace(0, 1, len(y_pred)), y_pred)

        # Generate null tracks
        null_hook_fractions = []
        null_hook_point_counts = []
        for perm in range(N_PERM):
            # Add Gaussian noise at the observed RMS level
            y_null = y_pred + RNG.normal(0, rms, n_pts)
            # Also randomly perturb x slightly to simulate measurement scatter
            x_null = x + RNG.normal(0, 0.02, n_pts)
            # Ensure monotonic ordering by radius (roughly)
            x_null = np.sort(x_null)[::-1] if np.mean(np.diff(x)) < 0 else np.sort(x_null)
            null_hf, null_hp, null_sc, _ = detect_hooks(x_null, y_null)
            null_hook_fractions.append(null_hf)
            null_hook_point_counts.append(null_hp)

        null_hook_fractions_all.extend(null_hook_fractions)

        # Compute p-value: fraction of null simulations with >= hook fraction
        null_arr = np.array(null_hook_fractions)
        p_value = np.mean(null_arr >= obs_hf)
        null_mean = np.mean(null_arr)
        null_std = np.std(null_arr)
        null_p95 = np.percentile(null_arr, 95)
        null_p99 = np.percentile(null_arr, 99)

        results.append({
            "galaxy": gal,
            "n_pts": n_pts,
            "hook_fraction": obs_hf,
            "hook_points": obs_hp,
            "sign_changes": obs_sc,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_p95": null_p95,
            "null_p99": null_p99,
            "p_value": p_value,
            "significant_95": obs_hf > null_p95,
            "significant_99": obs_hf > null_p99,
            "rms": rms,
        })

        if (g_idx + 1) % 50 == 0:
            print(f"    ... processed {g_idx+1}/{n_gal} galaxies")

    df_r = pd.DataFrame(results)
    n_fit = len(df_r)

    # Global statistics
    print(f"\n  {'='*60}")
    print(f"  RESULTS ({n_fit} galaxies with >=5 points)")
    print(f"  {'='*60}")

    print(f"\n  Observed:")
    print(f"    Galaxies with any hook: {n_with_hooks_observed}/{n_fit} "
          f"({100*n_with_hooks_observed/n_fit:.0f}%)")
    print(f"    Galaxies with significant hooks (>10%): {n_significant_hooks_observed}/{n_fit} "
          f"({100*n_significant_hooks_observed/n_fit:.0f}%)")
    print(f"    Mean hook fraction: {np.mean(observed_hook_fractions):.4f}")

    n_sig_95 = df_r["significant_95"].sum()
    n_sig_99 = df_r["significant_99"].sum()
    print(f"\n  Significant at 95% CL (vs null): {n_sig_95}/{n_fit} "
          f"({100*n_sig_95/n_fit:.0f}%)")
    print(f"  Significant at 99% CL (vs null): {n_sig_99}/{n_fit} "
          f"({100*n_sig_99/n_fit:.0f}%)")

    print(f"\n  Null distribution (over all N_perm * N_gal):")
    print(f"    Mean hook fraction: {np.mean(null_hook_fractions_all):.4f}")
    print(f"    P95: {np.percentile(null_hook_fractions_all, 95):.4f}")
    print(f"    P99: {np.percentile(null_hook_fractions_all, 99):.4f}")

    # How many galaxies show MORE hooks than null predicts?
    n_more_than_null = (df_r["hook_fraction"] > df_r["null_mean"]).sum()
    print(f"\n  Galaxies exceeding their null mean: {n_more_than_null}/{n_fit} "
          f"({100*n_more_than_null/n_fit:.0f}%)")

    # Expected number of galaxies exceeding 95% CL by chance: 5% * n_fit
    expected_95 = 0.05 * n_fit
    expected_99 = 0.01 * n_fit
    print(f"\n  Expected false positives by chance:")
    print(f"    At 95% CL: {expected_95:.0f} (observed: {n_sig_95})")
    print(f"    At 99% CL: {expected_99:.0f} (observed: {n_sig_99})")

    # Binomial test: is n_sig_95 significantly different from expected?
    from scipy.stats import binomtest
    bt_95 = binomtest(n_sig_95, n_fit, p=0.05, alternative="greater")
    bt_99 = binomtest(n_sig_99, n_fit, p=0.01, alternative="greater")
    print(f"\n  Binomial test (null = false positive rate):")
    print(f"    95% CL: n={n_sig_95}/{n_fit}, p={bt_95.pvalue:.4f}")
    print(f"    99% CL: n={n_sig_99}/{n_fit}, p={bt_99.pvalue:.4f}")

    # ── Plots ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Observed vs null mean hook fraction
    ax = axes[0, 0]
    ax.scatter(df_r["n_pts"], df_r["hook_fraction"], s=8, alpha=0.4, label="Observed")
    ax.scatter(df_r["n_pts"], df_r["null_mean"], s=8, alpha=0.4, color="red", label="Null mean")
    ax.set_xlabel("N points per galaxy")
    ax.set_ylabel("Hook fraction")
    ax.legend(fontsize=8)
    ax.set_title("(a) Observed vs Null Hook Fraction")

    # (b) P-value distribution
    ax = axes[0, 1]
    ax.hist(df_r["p_value"], bins=40, color="steelblue", edgecolor="white")
    ax.axvline(0.05, color="red", ls="--", lw=1.5, label="p=0.05")
    ax.axvline(0.01, color="darkred", ls=":", lw=1.5, label="p=0.01")
    ax.set_xlabel("P-value (null hypothesis)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.set_title("(b) P-value Distribution")

    # (c) Significant galaxies only
    ax = axes[0, 2]
    sig = df_r[df_r["significant_95"]]
    ax.scatter(sig["n_pts"], sig["hook_fraction"], s=20, alpha=0.6, color="darkred")
    for _, row in sig.iterrows():
        ax.annotate(row["galaxy"], (row["n_pts"], row["hook_fraction"]),
                    fontsize=5, alpha=0.7)
    ax.set_xlabel("N points")
    ax.set_ylabel("Hook fraction")
    ax.set_title(f"(c) Significant at 95% CL (n={len(sig)})")

    # (d) Hook fraction vs n_pts (observed histogram)
    ax = axes[1, 0]
    ax.hist(df_r["hook_fraction"], bins=30, color="steelblue", alpha=0.7, edgecolor="white",
            density=True, label="Observed")
    ax.hist(null_hook_fractions_all, bins=30, color="darkorange", alpha=0.5, edgecolor="white",
            density=True, label="Null")
    ax.axvline(0.1, color="red", ls="--", lw=1)
    ax.set_xlabel("Hook fraction")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title("(d) Observed vs Null Distribution")

    # (e) Hook fraction vs RMS
    ax = axes[1, 1]
    ax.scatter(df_r["rms"], df_r["hook_fraction"], s=8, alpha=0.4,
               c=df_r["n_pts"], cmap="viridis")
    ax.set_xlabel("In-track RMS (dex)")
    ax.set_ylabel("Hook fraction")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("N points")
    ax.set_title("(e) Hook Fraction vs Track Scatter")

    # (f) Hook fraction vs n_pts with significance coloring
    ax = axes[1, 2]
    colors = ["darkred" if s else "steelblue" for s in df_r["significant_95"]]
    alphas = [0.7 if s else 0.3 for s in df_r["significant_95"]]
    ax.scatter(df_r["n_pts"], df_r["hook_fraction"], s=15, c=colors, alpha=alphas)
    ax.set_xlabel("N points")
    ax.set_ylabel("Hook fraction")
    ax.set_title("(f) Significance (red = p<0.05)")

    plt.tight_layout()
    plt.savefig(f"{outdir}/hook_permutation_test.pdf", dpi=200)
    plt.savefig(f"{outdir}/hook_permutation_test.png", dpi=150)
    print(f"\n  Saved {outdir}/hook_permutation_test.png")
    plt.close()

    # Save results
    df_r.to_csv(f"{outdir}/hook_permutation_results.csv", index=False)
    print(f"  Saved {outdir}/hook_permutation_results.csv")

    # ── Top galaxies with real hooks ──
    print(f"\n  Top 20 galaxies by hook significance (lowest p-value):")
    top = df_r.sort_values("p_value").head(20)
    for _, row in top.iterrows():
        marker = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else "*")
        print(f"    {row['galaxy']:<15s} n={row['n_pts']:3d}  "
              f"hook_frac={row['hook_fraction']:.3f}  "
              f"p={row['p_value']:.4f} {marker}")

    return df_r


if __name__ == "__main__":
    df_results = run_null_test()
    print("\nPhase 3 hook permutation test complete.")

"""Phase 3 fix: GP-based hook null test.

Replace the parametric CPX5 null (which overfits small-N tracks) with a
Gaussian Process (RBF kernel) that preserves the smooth monotonic RAR
shape while properly sampling the covariance structure.

This fixes the adversary's Challenge 2: 28 galaxies with degenerate null.
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
N_PERM = 500

kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0


def load_data():
    df = parse_mass_models()
    df = df[df["R"] > 0].copy()
    Vbar_sq = (np.abs(df["Vgas"].values)*df["Vgas"].values
               + 0.5*df["Vdisk"].values**2 + 0.7*df["Vbul"].values**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"].values * kpc_to_m
    gbar = Vbar_sq * KM_S_TO_M_S**2 / R_m
    gobs = df["Vobs"].values**2 * KM_S_TO_M_S**2 / R_m
    valid = (gbar > 1e-13) & (gobs > 0)
    return df[valid].copy(), np.log10(gbar[valid]), np.log10(gobs[valid])


def detect_hooks(x, y):
    if len(x) < 5:
        return 0.0, 0, 0
    dx, dy = np.diff(x), np.diff(y)
    hook_points = sum(1 for i in range(len(dx)) if dx[i] < 0 and dy[i] > 0)
    y_smooth = np.convolve(y, np.ones(3)/3, mode="valid")
    dy_smooth = np.diff(y_smooth)
    sign_changes = sum(1 for i in range(1, len(dy_smooth))
                       if dy_smooth[i] * dy_smooth[i-1] < 0)
    return hook_points / max(len(x)-1, 1), hook_points, sign_changes


def run_gp_null_test(outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Phase 3 Fix: GP Hook Null Test")
    print(f"  Using Gaussian Process (RBF kernel) null model")
    print(f"  N_perm = {N_PERM}")
    print("=" * 60)

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except ImportError:
        print("  Installing scikit-learn...")
        import subprocess
        subprocess.check_call(["pip3", "install", "scikit-learn", "--break-system-packages", "-q"])
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    df, x_all, y_all = load_data()
    galaxies = sorted(df["ID"].unique())

    results = []
    print(f"\n  Processing {len(galaxies)} galaxies...")

    for g_idx, gal in enumerate(galaxies):
        sub = df[df["ID"] == gal].sort_values("R")
        x = sub["log_gbar"] = np.log10(
            (np.abs(sub["Vgas"])*sub["Vgas"] + 0.5*sub["Vdisk"]**2 + 0.7*sub["Vbul"]**2)
            * KM_S_TO_M_S**2 / (sub["R"] * kpc_to_m))
        x = np.maximum(x, np.log10(1e-13))
        y = np.log10(sub["Vobs"]**2 * KM_S_TO_M_S**2 / (sub["R"] * kpc_to_m))
        x, y = x.values, y.values

        n_pts = len(x)
        if n_pts < 5:
            continue

        # Observed
        obs_hf, obs_hp, obs_sc = detect_hooks(x, y)

        # GP null
        X = x.reshape(-1, 1)
        # RBF kernel with reasonable length scale for RAR (~1 dex in log-gbar)
        kernel = RBF(length_scale=1.0, length_scale_bounds=(0.3, 3.0)) + \
                 WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 0.5))
        try:
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                          alpha=1e-4, normalize_y=True)
            gp.fit(X, y)
        except Exception:
            # Fallback: fixed kernel
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)
            try:
                gp.fit(X, y)
            except Exception:
                continue

        # Generate null tracks from GP posterior
        null_hfs = []
        for _ in range(N_PERM):
            y_null = gp.sample_y(X, random_state=None).ravel()
            null_hf, _, _ = detect_hooks(x, y_null)
            null_hfs.append(null_hf)

        null_arr = np.array(null_hfs)
        null_mean = np.mean(null_arr)
        null_std = np.std(null_arr)
        null_p95 = np.percentile(null_arr, 95)
        p_value = np.mean(null_arr >= obs_hf) if null_std > 0 else 0.5
        is_degenerate = null_std < 1e-6

        results.append({
            "galaxy": gal, "n_pts": n_pts,
            "hook_fraction": obs_hf, "hook_points": obs_hp,
            "sign_changes": obs_sc,
            "null_mean": null_mean, "null_std": null_std,
            "null_p95": null_p95, "p_value": p_value,
            "significant_95": obs_hf > null_p95 if not is_degenerate else False,
            "degenerate_null": is_degenerate,
        })

        if (g_idx + 1) % 50 == 0:
            print(f"    ... {g_idx+1}/{len(galaxies)}")

    df_r = pd.DataFrame(results)
    n_fit = len(df_r)
    n_degen = df_r["degenerate_null"].sum()

    print(f"\n  {'='*60}")
    print(f"  GP NULL RESULTS ({n_fit} galaxies)")
    print(f"  Degenerate nulls (GP std=0): {n_degen} ({100*n_degen/n_fit:.0f}%)")
    print(f"  {'='*60}")

    # Only non-degenerate galaxies
    df_good = df_r[~df_r["degenerate_null"]]
    n_good = len(df_good)

    n_any_hook = (df_good["hook_fraction"] > 0).sum()
    n_sig = df_good["significant_95"].sum()
    expected_fp = 0.05 * n_good

    print(f"\n  Non-degenerate galaxies: {n_good}")
    print(f"  Any hook: {n_any_hook}/{n_good} ({100*n_any_hook/n_good:.0f}%)")
    print(f"  Significant at 95% CL: {n_sig}/{n_good} ({100*n_sig/n_good:.0f}%)")
    print(f"  Expected false positives: {expected_fp:.0f}")

    from scipy.stats import binomtest
    bt = binomtest(n_sig, n_good, p=0.05, alternative="greater")
    print(f"  Binomial test p = {bt.pvalue:.4f}")

    null_mean_all = df_good["null_mean"].mean()
    print(f"\n  GP null mean hook fraction: {null_mean_all:.4f}")
    print(f"  Observed mean hook fraction: {df_good['hook_fraction'].mean():.4f}")

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.scatter(df_good["n_pts"], df_good["hook_fraction"], s=8, alpha=0.4, label="Observed")
    ax.scatter(df_good["n_pts"], df_good["null_mean"], s=8, alpha=0.4, color="red", label="GP null mean")
    ax.set_xlabel("N points")
    ax.set_ylabel("Hook fraction")
    ax.legend(fontsize=8)
    ax.set_title(f"(a) GP Null vs Observed ({n_good} galaxies)")

    ax = axes[0, 1]
    ax.hist(df_good["p_value"], bins=40, color="steelblue", edgecolor="white")
    ax.axvline(0.05, color="red", ls="--", lw=1.5)
    ax.axvline(0.01, color="darkred", ls=":", lw=1.5)
    ax.set_xlabel("P-value (GP null)")
    ax.set_ylabel("Count")
    ax.set_title(f"(b) GP P-values (sig: {n_sig}, exp: {expected_fp:.0f})")

    ax = axes[1, 0]
    ax.hist(df_good["hook_fraction"], bins=30, color="steelblue", alpha=0.7, density=True,
            edgecolor="white", label="Observed")
    all_null = np.concatenate([np.array([r["null_mean"]]) for _, r in df_good.iterrows()])
    ax.hist(all_null, bins=30, color="darkorange", alpha=0.5, density=True,
            edgecolor="white", label="GP null (means)")
    ax.set_xlabel("Hook fraction")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title("(c) Hook Fraction Distributions")

    ax = axes[1, 1]
    sig_gals = df_good[df_good["significant_95"]]
    ax.scatter(sig_gals["n_pts"], sig_gals["hook_fraction"], s=20, alpha=0.6, color="darkred")
    for _, row in sig_gals.iterrows():
        ax.annotate(row["galaxy"], (row["n_pts"], row["hook_fraction"]), fontsize=5, alpha=0.7)
    ax.set_xlabel("N points")
    ax.set_ylabel("Hook fraction")
    ax.set_title(f"(d) GP-significant galaxies (n={len(sig_gals)})")

    plt.tight_layout()
    plt.savefig(f"{outdir}/gp_hook_null.pdf", dpi=200)
    plt.savefig(f"{outdir}/gp_hook_null.png", dpi=150)
    print(f"\n  Saved {outdir}/gp_hook_null.png")
    plt.close()

    # Compare with parametric null
    print(f"\n  Comparison with parametric CPX5 null:")
    print(f"    Parametric null: 28 sig, 28 degen, binomial p<1e-5")
    print(f"    GP null: {n_sig} sig, {n_degen} degen, binomial p={bt.pvalue:.4f}")
    if n_degen < 28:
        print(f"    → GP reduces degenerate nulls from 28 to {n_degen}")

    df_r.to_csv(f"{outdir}/gp_hook_null_results.csv", index=False)
    return df_r


if __name__ == "__main__":
    run_gp_null_test()
    print("\nDone.")

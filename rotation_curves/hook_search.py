"""Search individual SPARC RAR tracks for FIRE-2 'hook' features.

FIRE-2 (Ardizzone+2023) predicts non-monotonic RAR tracks ("hooks")
from cored dark matter halos in low-mass galaxies.
These cannot be explained by Modified Inertia MOND theories.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parse_sparc import parse_mass_models, compute_radial_accelerations


def detect_hooks(galaxy_track, min_hook_size=0.05):
    """Detect non-monotonic features in a galaxy's RAR track.

    A 'hook' is a region where gobs decreases while gbar decreases
    (i.e., the track doubles back on itself in the RAR plane).

    Returns: hook_score (float) — fraction of points showing non-monotonicity
    """
    x = galaxy_track["log_gbar"].values
    y = galaxy_track["log_gobs"].values

    if len(x) < 5:
        return 0.0, []

    # Sort by radius (should already be sorted, but ensure)
    # Check monotonicity: in the RAR, as we go outward, both gbar and gobs decrease
    # A hook occurs when this trend reverses locally

    # Check for local reversals: consecutive segments where slope changes sign
    dx = np.diff(x)
    dy = np.diff(y)

    # Smooth tracks (running mean of 3 points)
    y_smooth = np.convolve(y, np.ones(3)/3, mode="valid")
    x_smooth = np.convolve(x, np.ones(3)/3, mode="valid")
    dy_smooth = np.diff(y_smooth)
    dx_smooth = np.diff(x_smooth)

    # Count sign changes in the smoothed derivative
    sign_changes = 0
    for i in range(1, len(dy_smooth)):
        if dy_smooth[i] * dy_smooth[i-1] < 0:
            sign_changes += 1

    # Count points where gobs increases while going outward (gbar decreasing)
    # This is truly anomalous
    hook_points = 0
    for i in range(1, len(y)):
        if dx[i-1] < 0 and dy[i-1] > 0:  # gbar decreases, gobs increases
            hook_points += 1

    hook_fraction = hook_points / max(len(x) - 1, 1)

    return hook_fraction, {"sign_changes": sign_changes, "hook_points": hook_points}


def search_hooks(outdir="analysis"):
    """Search all SPARC galaxies for hook features."""
    print("=" * 60)
    print("Search for FIRE-2 Hook Features in SPARC RAR tracks")
    print("=" * 60)

    df = parse_mass_models()
    acc = compute_radial_accelerations(df)

    # Filter
    valid = np.isfinite(acc["log_gbar"]) & np.isfinite(acc["log_gobs"]) & (acc["gbar"] > 0)
    acc = acc[valid].copy()

    print(f"\n  Analyzing {acc['ID'].nunique()} galaxies, {len(acc)} points")

    # Per-galaxy hook analysis
    hook_results = []
    for gal in acc["ID"].unique():
        sub = acc[acc["ID"] == gal].sort_values("R")
        if len(sub) < 5:
            continue
        hook_frac, details = detect_hooks(sub)
        hook_results.append({
            "galaxy": gal,
            "n_pts": len(sub),
            "hook_fraction": hook_frac,
            "sign_changes": details["sign_changes"],
            "hook_points": details["hook_points"],
            "mean_resid": sub["log_gobs"].mean(),
        })

    df_hooks = pd.DataFrame(hook_results)

    # Statistics
    n_total = len(df_hooks)
    n_with_hooks = (df_hooks["hook_fraction"] > 0).sum()
    n_significant = (df_hooks["hook_fraction"] > 0.1).sum()
    print(f"\n  Total galaxies: {n_total}")
    print(f"  Galaxies with any hook: {n_with_hooks} ({100*n_with_hooks/n_total:.0f}%)")
    print(f"  Galaxies with significant hooks (>{10}%): {n_significant} ({100*n_significant/n_total:.0f}%)")
    print(f"  Mean hook fraction: {df_hooks['hook_fraction'].mean():.4f}")
    print(f"  Median hook points: {df_hooks['hook_points'].median():.0f}")

    # Top hook galaxies
    print(f"\n  Top 10 galaxies by hook fraction:")
    top = df_hooks.sort_values("hook_fraction", ascending=False).head(10)
    for _, row in top.iterrows():
        print(f"    {row['galaxy']:<15s} hook_frac={row['hook_fraction']:.3f} "
              f"n_pts={row['n_pts']}")

    # Plot examples
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    # Plot top hook galaxies
    for i in range(min(5, len(top))):
        gal = top.iloc[i]["galaxy"]
        sub = acc[acc["ID"] == gal].sort_values("R")
        ax = axes_flat[i]
        # Color by radius
        radii = sub["R"].values
        colors = plt.cm.viridis((radii - radii.min()) / max(radii.max() - radii.min(), 1))
        ax.scatter(sub["gbar"], sub["gobs"], c=colors, s=30, cmap="viridis", zorder=3)
        ax.plot(sub["gbar"], sub["gobs"], "k-", lw=0.5, alpha=0.5)
        # Annotate points with radius
        for j in range(len(sub)):
            if j % max(1, len(sub)//5) == 0:
                ax.annotate(f"{sub['R'].iloc[j]:.0f}kpc",
                           (sub['gbar'].iloc[j], sub['gobs'].iloc[j]),
                           fontsize=6, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("gbar (m/s²)"); ax.set_ylabel("gobs (m/s²)")
        ax.set_title(f"{gal} (hook={top.iloc[i]['hook_fraction']:.2f})")
        ax.set_xlim(1e-13, 1e-9); ax.set_ylim(1e-13, 1e-9)

        # Mark hook regions
        x, y = sub["log_gbar"].values, sub["log_gobs"].values
        for j in range(1, len(x)):
            if x[j] - x[j-1] < 0 and y[j] - y[j-1] > 0:
                ax.scatter(10**x[j], 10**y[j], s=80, facecolor="none",
                          edgecolor="r", linewidth=2, zorder=4)

    # Histogram
    ax = axes_flat[5]
    ax.hist(df_hooks["hook_fraction"], bins=30)
    ax.set_xlabel("Hook fraction")
    ax.set_ylabel("Count")
    ax.axvline(0.1, color="r", ls="--", label="significant threshold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/hook_search.png", dpi=150)
    print(f"\n  Saved {outdir}/hook_search.png")
    plt.close()

    # Plot global RAR colored by hook fraction
    fig, ax = plt.subplots(figsize=(10, 8))
    hook_map = dict(zip(df_hooks["galaxy"], df_hooks["hook_fraction"]))
    acc["hook_frac"] = acc["ID"].map(hook_map)

    scatter = ax.scatter(acc["gbar"], acc["gobs"], s=1, alpha=0.3,
                         c=acc["hook_frac"], cmap="RdYlBu_r", vmin=0, vmax=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gbar (m/s²)"); ax.set_ylabel("gobs (m/s²)")
    ax.set_title("SPARC RAR colored by hook fraction")
    plt.colorbar(scatter, ax=ax, label="Hook fraction")

    # MOND fit
    def mond_simple(gbar, a0):
        return gbar * (1 + np.sqrt(1 + 4*a0/np.maximum(gbar, 1e-20))) / 2
    gbar_model = np.logspace(-13, -9, 100)
    popt, _ = curve_fit(mond_simple, acc["gbar"].values, acc["gobs"].values,
                         p0=[1.2e-10], maxfev=10000)
    ax.plot(gbar_model, mond_simple(gbar_model, *popt), "r-", lw=2,
            label=f"MOND Simple a₀={popt[0]:.2e}")
    ax.legend()
    ax.set_xlim(1e-13, 1e-9); ax.set_ylim(1e-13, 1e-9)

    plt.tight_layout()
    plt.savefig(f"{outdir}/hook_colored_rar.png", dpi=150)
    print(f"  Saved {outdir}/hook_colored_rar.png")
    plt.close()

    # Save results
    df_hooks.to_csv(f"{outdir}/hook_search_results.csv", index=False)
    print(f"  Saved {outdir}/hook_search_results.csv")

    return df_hooks


if __name__ == "__main__":
    import os
    os.makedirs("analysis", exist_ok=True)
    df_hooks = search_hooks()
    print("\nDone.")

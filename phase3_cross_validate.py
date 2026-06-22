"""Phase 3 cross-validation: SPARC per-galaxy CPX5 (a,b) vs simulation predictions.

Test: do real SPARC galaxies' CPX5 parameters cluster near FIRE-2/baryonification,
as predicted by the simulation classifier? This independently validates the
Phase 3 simulation comparison using actual (not binned) galaxy-level data.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

OUTDIR = "analysis/phase3"
RNG = np.random.RandomState(42)

# Simulation CPX5 bootstrap means ± std (from phase3_simulations.py)
SIM_RESULTS = {
    "EAGLE":         {"a": -16.32, "b": -66.72, "a_std": 0.02, "b_std": 0.21},
    "IllustrisTNG":  {"a": -16.48, "b": -67.58, "a_std": 0.02, "b_std": 0.25},
    "FIRE-2":        {"a": -16.88, "b": -71.13, "a_std": 0.02, "b_std": 0.18},
    "MassiveBlack-II": {"a": -18.72, "b": -88.90, "a_std": 0.09, "b_std": 0.97},
    "Baryonification": {"a": -16.91, "b": -71.51, "a_std": 0.02, "b_std": 0.20},
}

SIM_COLORS = {
    "EAGLE": "orange",
    "IllustrisTNG": "green",
    "FIRE-2": "red",
    "MassiveBlack-II": "purple",
    "Baryonification": "brown",
}


def cross_validate(outdir=OUTDIR):
    import os
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Phase 3: SPARC Cross-validation vs Simulations")
    print("=" * 60)

    # Load per-galaxy CPX5 params
    df_pg = pd.read_csv("rotation_curves/analysis/per_galaxy_cpx5_params.csv")
    print(f"  Loaded {len(df_pg)} galaxies with per-galaxy CPX5 fits")
    print(f"  CPX5 a: {df_pg['a'].mean():.2f} ± {df_pg['a'].std():.2f}")
    print(f"  CPX5 b: {df_pg['b'].mean():.2f} ± {df_pg['b'].std():.2f}")

    # Filter to well-constrained fits (a_err / |a| < 0.5, b_err / |b| < 0.5)
    df_well = df_pg[(df_pg["e_a"] / np.abs(df_pg["a"]) < 0.5) &
                     (df_pg["e_b"] / np.abs(df_pg["b"]) < 0.5)].copy()
    print(f"  Well-constrained galaxies (err/|param| < 0.5): {len(df_well)}")

    # Distance of each galaxy to each simulation
    for sim, params in SIM_RESULTS.items():
        da = df_well["a"] - params["a"]
        db = df_well["b"] - params["b"]
        df_well[f"dist_{sim}"] = np.sqrt(da**2 + (db/10)**2)

    # Assign each galaxy to closest simulation
    dist_cols = [f"dist_{sim}" for sim in SIM_RESULTS]
    df_well["closest_sim"] = df_well[dist_cols].idxmin(axis=1).str.replace("dist_", "")

    print(f"\n  Galaxy assignment to simulations:")
    for sim in SIM_RESULTS:
        n = (df_well["closest_sim"] == sim).sum()
        print(f"    {sim:<20s}: {n:3d} galaxies ({100*n/len(df_well):.0f}%)")

    # ── Figure ───────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (a) CPX5 (a,b) space: SPARC galaxies vs simulations
    ax = axes[0, 0]

    # Simulation 2σ ellipses
    for sim, params in SIM_RESULTS.items():
        a_c, b_c = params["a"], params["b"]
        a_s, b_s = params["a_std"] * 3, params["b_std"] * 3  # 3σ for visibility
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(a_c + a_s*np.cos(theta), b_c + b_s*np.sin(theta),
                color=SIM_COLORS[sim], lw=2, alpha=0.5, ls="--")
        ax.scatter([a_c], [b_c], c=SIM_COLORS[sim], s=100,
                   edgecolors="k", linewidth=0.8, zorder=5, label=sim)

    # SPARC galaxies
    sc = ax.scatter(df_well["a"], df_well["b"], s=8, alpha=0.4,
                    c=df_well["rms"], cmap="RdYlBu_r", edgecolors="none")

    # KDE of SPARC galaxies
    try:
        a_vals, b_vals = df_well["a"].values, df_well["b"].values
        kde = gaussian_kde(np.vstack([a_vals, b_vals]).T)
        a_grid_kde = np.linspace(a_vals.min(), a_vals.max(), 60)
        b_grid_kde = np.linspace(b_vals.min(), b_vals.max(), 60)
        A_K, B_K = np.meshgrid(a_grid_kde, b_grid_kde)
        Z = kde(np.vstack([A_K.ravel(), B_K.ravel()])).reshape(A_K.shape)
        ax.contour(A_K, B_K, Z, levels=[np.max(Z)*np.exp(-2.0)], colors="blue",
                   linewidths=2, linestyles="-")
    except Exception:
        pass

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Per-galaxy RMS (dex)")
    ax.set_xlabel("CPX5 a")
    ax.set_ylabel("CPX5 b")
    ax.set_title(f"(a) SPARC ({len(df_well)} galaxies) vs Simulation Predictions")
    ax.legend(fontsize=7, loc="upper left")

    # (b) Histogram of closest simulation
    ax = axes[0, 1]
    sim_order = ["FIRE-2", "Baryonification", "IllustrisTNG", "EAGLE", "MassiveBlack-II"]
    counts = [sum(df_well["closest_sim"] == s) for s in sim_order]
    colors_bar = [SIM_COLORS[s] for s in sim_order]
    bars = ax.bar(range(len(sim_order)), counts, color=colors_bar, edgecolor="k")
    ax.set_xticks(range(len(sim_order)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in sim_order], fontsize=8)
    ax.set_ylabel("Number of SPARC galaxies")
    ax.set_title("(b) Closest Matching Simulation")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha="center", fontsize=9, fontweight="bold")

    # (c) CPX5 b vs n_pts (visibility)
    ax = axes[1, 0]
    for sim in sim_order:
        mask = df_well["closest_sim"] == sim
        ax.scatter(df_well.loc[mask, "n_pts"], df_well.loc[mask, "b"],
                   c=SIM_COLORS[sim], s=15, alpha=0.5, edgecolors="none")
    ax.set_xlabel("N points in rotation curve")
    ax.set_ylabel("CPX5 b")
    ax.set_title("(c) CPX5 b vs Track Length")
    # Add simulation b values
    for sim in sim_order:
        ax.axhline(SIM_RESULTS[sim]["b"], color=SIM_COLORS[sim], ls=":", lw=1, alpha=0.5)

    # (d) Distance to FIRE-2 vs gas fraction
    ax = axes[1, 1]
    df_well["dist_fire2"] = df_well["dist_FIRE-2"]
    sc = ax.scatter(df_well["gas_frac"], df_well["dist_fire2"],
                    s=15, alpha=0.5, c=df_well["rms"], cmap="RdYlBu_r")
    ax.set_xlabel("Gas fraction")
    ax.set_ylabel("Distance to FIRE-2 in CPX5 space")
    ax.set_title("(d) Distance to FIRE-2 vs Gas Fraction")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("RMS (dex)")
    # Correlation
    from scipy.stats import spearmanr
    r, p = spearmanr(df_well["gas_frac"], df_well["dist_fire2"])
    ax.text(0.95, 0.95, f"ρ={r:.3f}, p={p:.2e}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{outdir}/sparc_cross_validation.pdf", dpi=200)
    plt.savefig(f"{outdir}/sparc_cross_validation.png", dpi=150)
    print(f"\n  Saved {outdir}/sparc_cross_validation.png")
    plt.close()

    # Summary
    fire2_count = (df_well["closest_sim"] == "FIRE-2").sum()
    bary_count = (df_well["closest_sim"] == "Baryonification").sum()
    print(f"\n  {'='*60}")
    print(f"  CROSS-VALIDATION RESULT")
    print(f"  {'='*60}")
    print(f"  SPARC galaxies closest to FIRE-2: {fire2_count} ({100*fire2_count/len(df_well):.0f}%)")
    print(f"  SPARC galaxies closest to Baryonification: {bary_count} ({100*bary_count/len(df_well):.0f}%)")
    print(f"  Combined (FIRE-2 + Baryonification): {fire2_count + bary_count} "
          f"({100*(fire2_count+bary_count)/len(df_well):.0f}%)")

    total = fire2_count + bary_count
    n = len(df_well)
    from scipy.stats import binomtest
    bt = binomtest(total, n, p=2/5, alternative="greater")
    print(f"  Binomial test vs uniform (2/5 models): p = {bt.pvalue:.4f}")

    return df_well


if __name__ == "__main__":
    cross_validate()
    print("\nDone.")

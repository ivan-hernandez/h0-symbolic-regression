"""Test whether MOND a₀ correlates with galaxy properties.

Literature says a₀ is "universal" but this has never been tested
with per-galaxy fits to a large homogeneous sample. We compute a₀
for each SPARC galaxy and correlate with gas fraction, surface
brightness, V_max, distance, and the CPX5 b parameter.
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotation_curves"))
sys.path.insert(0, ".")
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

OUTDIR = "analysis/a0_properties"
os.makedirs(OUTDIR, exist_ok=True)

kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0


def mond_simple(gbar, a0):
    return gbar * (1 + np.sqrt(1 + 4*a0/np.maximum(gbar, 1e-20))) / 2


def mond_mcgaugh(gbar, a0):
    return gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20)/a0)), 1e-20)


def cpx5_log(x, a, b):
    return a + b / np.maximum(x, -50)


def fit_per_galaxy(df):
    """Fit MOND Simple, MOND McGaugh, and CPX5 to each galaxy."""
    gal_ids = df["ID"].unique()
    results = []
    for gal in gal_ids:
        sub = df[df["ID"] == gal].sort_values("R")
        if len(sub) < 5:
            continue
        x_log = sub["log_gbar"].values
        y_log = sub["log_gobs"].values
        gbar = sub["gbar"].values
        gobs = sub["gobs"].values

        # MOND Simple
        try:
            popt_s, _ = curve_fit(mond_simple, gbar, gobs, p0=[1.2e-10],
                                   maxfev=10000, bounds=(1e-12, 1e-8))
            a0_simple = popt_s[0]
            pred_s = mond_simple(gbar, a0_simple)
            rms_s = np.sqrt(np.mean((np.log10(gobs) - np.log10(pred_s))**2))
        except:
            a0_simple = np.nan
            rms_s = np.nan

        # MOND McGaugh
        try:
            popt_m, _ = curve_fit(mond_mcgaugh, gbar, gobs, p0=[1.2e-10],
                                   maxfev=10000, bounds=(1e-12, 1e-8))
            a0_mcg = popt_m[0]
            pred_m = mond_mcgaugh(gbar, a0_mcg)
            rms_m = np.sqrt(np.mean((np.log10(gobs) - np.log10(pred_m))**2))
        except:
            a0_mcg = np.nan
            rms_m = np.nan

        # CPX5
        try:
            popt_c, _ = curve_fit(cpx5_log, x_log, y_log, p0=[-17, -70], maxfev=10000)
            a_c5, b_c5 = popt_c
            pred_c = cpx5_log(x_log, *popt_c)
            rms_c = np.sqrt(np.mean((y_log - pred_c)**2))
        except:
            a_c5, b_c5 = np.nan, np.nan
            rms_c = np.nan

        # Galaxy properties
        gas_frac = np.mean(np.abs(sub["Vgas"]) / np.maximum(sub["Vobs"], 0.1))
        sb_total = np.median(sub["SBdisk"] + sub["SBbul"])
        log_sb = np.log10(np.maximum(sb_total, 0.1))
        v_max = sub["Vobs"].max()
        d_mpc = sub["D"].iloc[0]
        n_pts = len(sub)

        # Mass-based gas fraction
        gas_mass = np.abs(sub["Vgas"]).mean()
        disk_mass = 0.5 * (sub["Vdisk"]**2).mean()
        bulge_mass = 0.7 * (sub["Vbul"]**2).mean()
        f_gas_mass = gas_mass / max(gas_mass + disk_mass + bulge_mass, 1e-10)

        results.append({
            "galaxy": gal, "n_pts": n_pts,
            "a0_simple": a0_simple, "rms_simple": rms_s,
            "a0_mcgaugh": a0_mcg, "rms_mcgaugh": rms_m,
            "cpx5_a": a_c5, "cpx5_b": b_c5, "rms_cpx5": rms_c,
            "gas_frac": f_gas_mass,
            "log_SB": log_sb,
            "V_max": v_max, "D_mpc": d_mpc,
        })

    df_r = pd.DataFrame(results)
    return df_r


def run_a0_analysis(outdir=OUTDIR):
    print("=" * 60)
    print("MOND a₀ — Galaxy Property Dependence")
    print("=" * 60)

    df = parse_mass_models()
    df = df[df["R"] > 0].copy()
    Vbar_sq = (np.abs(df["Vgas"])*df["Vgas"]
               + 0.5*df["Vdisk"]**2 + 0.7*df["Vbul"]**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"] * kpc_to_m
    df["gbar"] = Vbar_sq * KM_S_TO_M_S**2 / R_m
    df["gobs"] = df["Vobs"]**2 * KM_S_TO_M_S**2 / R_m
    valid = (df["gbar"] > 1e-13) & (df["gobs"] > 0)
    df = df[valid].copy()
    df["log_gbar"] = np.log10(df["gbar"])
    df["log_gobs"] = np.log10(df["gobs"])

    print(f"\n  Fitting {df['ID'].nunique()} galaxies...")
    df_r = fit_per_galaxy(df)

    # Filter to well-fit galaxies
    good = df_r["rms_mcgaugh"].notna() & (df_r["rms_mcgaugh"] < 0.5)
    df_g = df_r[good].copy()
    print(f"  Well-fit galaxies (MOND McGaugh RMS < 0.5 dex): {len(df_g)}")

    # Statistics
    a0_med = np.median(df_g["a0_mcgaugh"])
    a0_p16, a0_p84 = np.percentile(df_g["a0_mcgaugh"], [16, 84])
    print(f"\n  MOND McGaugh a₀:")
    print(f"    Median: {a0_med:.3e} m/s²")
    print(f"    68% CL: [{a0_p16:.3e}, {a0_p84:.3e}]")
    print(f"    Ratio max/min: {df_g['a0_mcgaugh'].max()/df_g['a0_mcgaugh'].min():.1f}×")
    print(f"    Canonical a₀ = 1.2e-10")
    print(f"    Fraction within factor 2 of canonical: "
          f"{(np.abs(np.log10(df_g['a0_mcgaugh']/1.2e-10)) < np.log10(2)).mean():.0%}")

    # Correlations
    props = {"gas_frac": "Gas fraction", "log_SB": "log SB",
             "V_max": "V_max [km/s]", "D_mpc": "Distance [Mpc]"}
    print(f"\n  Spearman correlations (a₀ vs property):")
    print(f"  {'Property':<20s} {'ρ (a₀)':<10s} {'p':<12s} {'ρ (CPX5 b)':<12s} {'p':<12s}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    for col, label in props.items():
        r_a0, p_a0 = spearmanr(df_g[col], df_g["a0_mcgaugh"])
        r_c5, p_c5 = spearmanr(df_g[col], df_g["cpx5_b"])
        print(f"  {label:<20s} {r_a0:+.4f}     {p_a0:<12.2e} {r_c5:+.4f}     {p_c5:<12.2e}")

    # Best predictor
    abs_r_a0 = [abs(spearmanr(df_g[c], df_g["a0_mcgaugh"])[0]) for c in props]
    best_prop = list(props.values())[np.argmax(abs_r_a0)]
    print(f"\n  Strongest a₀ correlation: {best_prop} (|ρ|={max(abs_r_a0):.4f})")

    # ── Figure ───────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for i, (col, label) in enumerate(props.items()):
        ax = axes[i // 3][i % 3]
        sc = ax.scatter(df_g[col], df_g["a0_mcgaugh"], s=10, alpha=0.5,
                        c=df_g["rms_mcgaugh"], cmap="RdYlBu_r")
        ax.set_xscale("log" if col in ["a0_mcgaugh", "V_max"] else "linear")
        ax.set_yscale("log")
        r, p = spearmanr(df_g[col], df_g["a0_mcgaugh"])
        ax.set_xlabel(label)
        ax.set_ylabel("a₀ (MOND McGaugh) [m/s²]")
        ax.axhline(1.2e-10, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(f"ρ={r:+.3f} (p={p:.2e})")
        if i == 0:
            plt.colorbar(sc, ax=ax, label="RMS (dex)")

    # (e) a₀ histogram
    ax = axes[1, 2]
    ax.hist(df_g["a0_mcgaugh"], bins=30, color="steelblue", edgecolor="white")
    ax.axvline(a0_med, color="k", ls="-", lw=1.5)
    ax.axvline(1.2e-10, color="red", ls="--", lw=1.5, label="Canonical a₀")
    ax.set_xscale("log")
    ax.set_xlabel("a₀ [m/s²]")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.set_title(f"a₀ distribution (×{df_g['a0_mcgaugh'].max()/df_g['a0_mcgaugh'].min():.0f} range)")

    plt.tight_layout()
    plt.savefig(f"{outdir}/a0_properties.pdf", dpi=200)
    plt.savefig(f"{outdir}/a0_properties.png", dpi=150)
    print(f"\n  Saved {outdir}/a0_properties.png")
    plt.close()

    # Compare a₀ scatter vs CPX5 scatter
    print(f"\n  {'='*60}")
    print(f"  STABILITY COMPARISON")
    print(f"  {'='*60}")
    a0_spread = df_g["a0_mcgaugh"].std() / df_g["a0_mcgaugh"].mean()
    c5_spread = abs(df_g["cpx5_b"].std() / df_g["cpx5_b"].mean())
    print(f"  MOND a₀ fractional scatter: {a0_spread:.1%}")
    print(f"  CPX5 b fractional scatter:   {c5_spread:.1%}")
    print(f"  → a₀ is {a0_spread/c5_spread:.1f}× more scattered than CPX5 b")

    # Save
    df_r.to_csv(f"{outdir}/a0_per_galaxy.csv", index=False)
    print(f"  Saved {outdir}/a0_per_galaxy.csv")

    return df_r


if __name__ == "__main__":
    run_a0_analysis()
    print("\nDone.")

"""Cross-validate CPX5 on SPARC sub-samples: dwarfs vs spirals vs LSB.

If CPX5 is universal, different morphological types should produce
consistent (a,b) parameters. This is an internal cross-validation
using only SPARC data — no new data download needed.
"""
import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(script_dir, "rotation_curves"))
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parse_sparc import parse_mass_models
import os, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/phase3"
os.makedirs(OUTDIR, exist_ok=True)
kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0

def cpx5_log(x, a, b):
    return a + b / np.maximum(x, -50)

def load_data():
    df = parse_mass_models()
    df = df[df["R"] > 0].copy()
    Vbar_sq = (np.abs(df["Vgas"])*df["Vgas"]
               + 0.5*df["Vdisk"]**2 + 0.7*df["Vbul"]**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"] * kpc_to_m
    gbar = Vbar_sq * KM_S_TO_M_S**2 / R_m
    gobs = df["Vobs"]**2 * KM_S_TO_M_S**2 / R_m
    valid = (gbar > 1e-13) & (gobs > 0)
    df = df[valid].copy()
    df["log_gbar"] = np.log10(gbar[valid])
    df["log_gobs"] = np.log10(gobs[valid])
    df["gbar"] = gbar[valid]
    df["gobs"] = gobs[valid]
    return df

def classify_galaxies(df):
    """Classify galaxies by mass, gas fraction, and surface brightness."""
    gal_props = df.groupby("ID").agg(
        V_max=("Vobs", "max"),
        log_gbar_med=("log_gbar", "median"),
        SBdisk=("SBdisk", "median"),
        D=("D", "first"),
        n_pts=("R", "count"),
    ).reset_index()

    # Gas fraction using velocity ratios (proper mass-based)
    f_gas_vals = {}
    for gal in df["ID"].unique():
        sub = df[df["ID"] == gal]
        gas = np.abs(sub["Vgas"]).mean()
        disk = np.sqrt(np.abs(sub["Vdisk"])).mean()**2
        bulge = np.sqrt(np.abs(sub["Vbul"])).mean()**2
        total = gas + 0.5*disk + 0.7*bulge
        f_gas_vals[gal] = gas / max(total, 1e-10)
    gal_props["f_gas"] = gal_props["ID"].map(f_gas_vals)

    # Classifications
    gal_props["type"] = "spiral"
    gal_props.loc[gal_props["V_max"] < 80, "type"] = "dwarf"
    gal_props.loc[gal_props["V_max"] > 180, "type"] = "massive"

    gal_props["gas_class"] = "gas-poor"
    gal_props.loc[gal_props["f_gas"] > 0.3, "gas_class"] = "gas-rich"

    gal_props["SB_class"] = "HSB"
    gal_props.loc[gal_props["SBdisk"] < 0.5, "SB_class"] = "LSB"

    return gal_props


def run_crossval(outdir=OUTDIR):
    print("=" * 60)
    print("Phase 3: SPARC Sub-Sample Cross-Validation")
    print("=" * 60)

    df = load_data()
    gal_props = classify_galaxies(df)
    print(f"\n  Galaxy classification:")
    for t in ["dwarf", "spiral", "massive"]:
        n = (gal_props["type"] == t).sum()
        print(f"    {t:<10s}: {n:3d} galaxies")
    for g in ["gas-poor", "gas-rich"]:
        n = (gal_props["gas_class"] == g).sum()
        print(f"    {g:<10s}: {n:3d} galaxies")

    # Fit CPX5 to each sub-sample
    results = {}
    classifications = {
        "All galaxies": df["ID"].unique(),
        "Dwarfs (V<80)": gal_props[gal_props["type"]=="dwarf"]["ID"],
        "Spirals (80<V<180)": gal_props[gal_props["type"]=="spiral"]["ID"],
        "Massive (V>180)": gal_props[gal_props["type"]=="massive"]["ID"],
        "Gas-poor": gal_props[gal_props["gas_class"]=="gas-poor"]["ID"],
        "Gas-rich": gal_props[gal_props["gas_class"]=="gas-rich"]["ID"],
        "HSB": gal_props[gal_props["SB_class"]=="HSB"]["ID"],
        "LSB": gal_props[gal_props["SB_class"]=="LSB"]["ID"],
    }

    print(f"\n  {'Sample':<25s} {'N_gal':<8s} {'N_pts':<8s} {'a':<10s} {'b':<10s} {'RMS':<8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    samples = []
    for name, gals in classifications.items():
        sub = df[df["ID"].isin(gals)]
        x, y = sub["log_gbar"].values, sub["log_gobs"].values
        try:
            popt, pcov = curve_fit(cpx5_log, x, y, p0=[-17, -70], maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            pred = cpx5_log(x, *popt)
            rms = np.sqrt(np.mean((y - pred)**2))
        except:
            popt, perr = [np.nan, np.nan], [np.nan, np.nan]
            rms = np.nan

        print(f"  {name:<25s} {sub['ID'].nunique():<8d} {len(sub):<8d} "
              f"{popt[0]:<10.2f} {popt[1]:<10.2f} {rms:<8.4f}")

        results[name] = {"a": popt[0], "b": popt[1], "a_err": perr[0], "b_err": perr[1],
                         "rms": rms, "n_gal": sub["ID"].nunique(), "n_pts": len(sub)}
        samples.append({"name": name, "a": popt[0], "b": popt[1],
                        "a_err": perr[0], "b_err": perr[1]})

    # Consistency check
    global_a, global_b = results["All galaxies"]["a"], results["All galaxies"]["b"]
    print(f"\n  Deviation from global fit (a={global_a:.2f}, b={global_b:.2f}):")
    max_dev = 0
    for name in classifications:
        if name == "All galaxies":
            continue
        r = results[name]
        da, db = r["a"] - global_a, r["b"] - global_b
        sig_a = abs(da) / max(r["a_err"], 0.01)
        sig_b = abs(db) / max(r["b_err"], 0.1)
        max_dev = max(max_dev, sig_a, sig_b)
        print(f"    {name:<25s}: Δa={da:+.2f} ({sig_a:.1f}σ)  Δb={db:+.0f} ({sig_b:.1f}σ)")

    universal = max_dev < 3
    print(f"\n  Max deviation: {max_dev:.1f}σ — {'UNIVERSAL: all sub-samples consistent' if universal else 'TENSION: sub-samples disagree'}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    colors_simple = {"All galaxies": "black",
                     "Dwarfs (V<80)": "red", "Spirals (80<V<180)": "blue",
                     "Massive (V>180)": "green"}
    x_grid = np.linspace(-13, -8, 300)
    for name in ["Dwarfs (V<80)", "Spirals (80<V<180)", "Massive (V>180)"]:
        ax.plot(x_grid, cpx5_log(x_grid, results[name]["a"], results[name]["b"]),
                color=colors_simple[name], lw=2, label=f"{name} (a={results[name]['a']:.1f})")
    ax.plot(x_grid, cpx5_log(x_grid, global_a, global_b),
            "k-", lw=3, alpha=0.3, label="All galaxies")
    ax.plot(x_grid, x_grid, "k:", lw=0.5, alpha=0.2)
    ax.set_xlabel("log g_bar [m/s²]")
    ax.set_ylabel("log g_obs [m/s²]")
    ax.set_title("(a) CPX5 by Morphological Type")
    ax.legend(fontsize=8)
    ax.set_xlim(-13, -8)
    ax.set_ylim(-13, -8)

    ax = axes[1]
    for s in samples:
        ax.errorbar(s["a"], s["b"], xerr=s["a_err"], yerr=s["b_err"],
                    fmt="o", ms=8, capsize=3, label=s["name"][:20])
    ax.set_xlabel("CPX5 a")
    ax.set_ylabel("CPX5 b")
    ax.set_title("(b) CPX5 Parameters by Sub-Sample")
    ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{outdir}/subsample_crossval.pdf", dpi=200)
    plt.savefig(f"{outdir}/subsample_crossval.png", dpi=150)
    print(f"\n  Saved {outdir}/subsample_crossval.png")
    plt.close()

    # Save
    pd.DataFrame(samples).to_csv(f"{outdir}/subsample_cpx5_params.csv", index=False)

    return results


if __name__ == "__main__":
    run_crossval()
    print("\nDone.")

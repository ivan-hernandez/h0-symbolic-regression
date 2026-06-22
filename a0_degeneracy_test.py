"""Test: Is MOND a₀ actually measurable per-galaxy, or degenerate?

For 10 representative galaxies, compute chi2(a₀) profiles to see
if a₀ is constrained or unconstrained by individual rotation curves.
"""
import numpy as np
from scipy.optimize import curve_fit
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotation_curves"))
sys.path.insert(0, ".")
from parse_sparc import parse_mass_models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

OUTDIR = "analysis/a0_properties"
os.makedirs(OUTDIR, exist_ok=True)

kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0


def mond_mcgaugh(gbar, a0):
    return gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20)/a0)), 1e-20)


def load_data():
    df = parse_mass_models()
    df = df[df["R"] > 0].copy()
    Vbar_sq = (np.abs(df["Vgas"])*df["Vgas"] + 0.5*df["Vdisk"]**2 + 0.7*df["Vbul"]**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"] * kpc_to_m
    df["gbar"] = Vbar_sq * KM_S_TO_M_S**2 / R_m
    df["gobs"] = df["Vobs"]**2 * KM_S_TO_M_S**2 / R_m
    valid = (df["gbar"] > 1e-13) & (df["gobs"] > 0)
    return df[valid].copy()


df = load_data()

# Select 10 galaxies with a range of properties
gals = df.groupby("ID").agg(n=("R", "count"), V=("Vobs", "max")).reset_index()
gals = gals[gals["n"] >= 10].sort_values("V")
# Take every ~17th galaxy for diversity
sample_gals = gals.iloc[::17]["ID"].values[:10]

a0_grid = np.logspace(-12, -9, 100)

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for i, gal in enumerate(sample_gals):
    sub = df[df["ID"] == gal].sort_values("R")
    gbar, gobs = sub["gbar"].values, sub["gobs"].values
    sigma = np.maximum(0.1 * gobs, sub["e_Vobs"].values * 2 * sub["Vobs"].values
                       * KM_S_TO_M_S**2 / (sub["R"].values * kpc_to_m))

    # Chi2 profile
    chi2_vals = []
    for a0 in a0_grid:
        pred = mond_mcgaugh(gbar, a0)
        chi2 = np.sum(((gobs - pred) / sigma)**2)
        chi2_vals.append(chi2)

    chi2_arr = np.array(chi2_vals)
    dchi2 = chi2_arr - chi2_arr.min()

    ax = axes[i]
    ax.semilogx(a0_grid, dchi2, "b-", lw=1.5)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5, label="Δχ²=1 (68% CL)")
    ax.axvline(1.2e-10, color="k", ls=":", lw=0.8, alpha=0.5)

    # Find 68% interval
    within1 = a0_grid[dchi2 <= 1.0]
    if len(within1) > 0:
        a0_lo, a0_hi = within1[0], within1[-1]
        ax.axvspan(a0_lo, a0_hi, alpha=0.1, color="blue")
        constraint = f"[{a0_lo:.1e}, {a0_hi:.1e}]"
        spread = np.log10(a0_hi / a0_lo)
    else:
        constraint = "unconstrained"
        spread = 99

    ax.set_title(f"{gal} (n={len(sub)})", fontsize=8)
    if i >= 5:
        ax.set_xlabel("a₀ [m/s²]")
    ax.set_ylabel("Δχ²" if i % 5 == 0 else "")

    # Constraint level
    color = "green" if spread < 0.5 else ("orange" if spread < 1.0 else "red")
    ax.text(0.95, 0.95, f"a₀ {constraint}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=color, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/a0_degeneracy.pdf", dpi=200)
plt.savefig(f"{OUTDIR}/a0_degeneracy.png", dpi=150)
print(f"Saved {OUTDIR}/a0_degeneracy.png")

# Count constrained vs unconstrained
print(f"\nPer-galaxy a₀ constraint summary:")
for gal in sample_gals:
    sub = df[df["ID"] == gal]
    gbar, gobs = sub["gbar"].values, sub["gobs"].values
    sigma = np.maximum(0.1*gobs, 0.01)
    chi2_arr = np.array([np.sum(((gobs - mond_mcgaugh(gbar, a0))/sigma)**2)
                         for a0 in a0_grid])
    dchi2 = chi2_arr - chi2_arr.min()
    within1 = a0_grid[dchi2 <= 1.0]
    if len(within1) > 0:
        spread_dex = np.log10(within1[-1] / within1[0])
        status = "constrained" if spread_dex < 0.5 else "weakly constrained"
    else:
        spread_dex = 99
        status = "UNCONSTRAINED"
    print(f"  {gal:<15s} n={len(sub):3d}  a₀ range={spread_dex:.1f} dex  → {status}")

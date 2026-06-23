"""DM profiles: direct ρ(r) for best-measured SPARC galaxies.

Uses the 20 galaxies with most rotation curve points and
compares inner density slopes: cusp (NFW: β≈−1) vs core (β≈0).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parse_sparc import parse_mass_models
import os, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/dm_profiles"
os.makedirs(OUTDIR, exist_ok=True)

kpc_to_m = 3.0857e19
G_SI = 6.6743e-11
Msun_kg = 1.989e30

df = parse_mass_models()
df = df[df["R"] > 0].copy()

# Get top 20 galaxies by N points
gal_counts = df.groupby("ID").size().sort_values(ascending=False)
top_gals = gal_counts.head(20).index

print("=" * 60)
print("DM Inner Slopes — Best-Measured SPARC Galaxies")
print("=" * 60)

results = []
for gal in top_gals:
    sub = df[df["ID"] == gal].sort_values("R")
    if len(sub) < 10:
        continue

    R_kpc = sub["R"].values
    V_c = sub["Vobs"].values
    V_gas = np.abs(sub["Vgas"]).values
    V_star = np.sqrt(0.5*sub["Vdisk"]**2 + 0.7*sub["Vbul"]**2)

    # Enclosed mass: M(R) = V²R/G
    R_m = R_kpc * kpc_to_m
    M_tot = V_c**2 * 1e6 * R_m / G_SI / Msun_kg
    M_bar = (V_gas**2 + V_star**2) * 1e6 * R_m / G_SI / Msun_kg
    M_dm = np.maximum(M_tot - M_bar, 1e3)

    # ρ(R) from ΔM/ΔV in a sphere
    dM = np.gradient(M_dm)
    dR = np.gradient(R_kpc)
    rho = np.abs(dM / dR) / (4 * np.pi * R_kpc**2)  # Msun/kpc³

    # Inner slope (R < 2 kpc)
    inner = R_kpc < 2.0
    if inner.sum() >= 3:
        log_r_inner = np.log10(R_kpc[inner])
        log_rho_inner = np.log10(np.maximum(rho[inner], 1e-5))
        slope, intercept = np.polyfit(log_r_inner, log_rho_inner, 1)
    else:
        slope = np.nan

    V_max = V_c.max()
    D_mpc = sub["D"].iloc[0]
    results.append({
        "galaxy": gal, "n_pts": len(sub),
        "inner_slope": slope, "V_max": V_max,
        "D": D_mpc,
    })

    print(f"  {gal:<12s} n={len(sub):3d}  β_inner={slope:+5.2f}  V_max={V_max:6.0f}")

df_r = pd.DataFrame(results).dropna()
print(f"\n  Inner slope distribution ({len(df_r)} galaxies):")
print(f"    Mean: {df_r['inner_slope'].mean():+.2f}")
print(f"    Median: {df_r['inner_slope'].median():+.2f}")
print(f"    Std: {df_r['inner_slope'].std():.2f}")

n_cusp = (df_r["inner_slope"] < -0.5).sum()
n_core = (df_r["inner_slope"] > -0.5).sum()
print(f"\n    Cusp (β < −0.5): {n_cusp} galaxies ({100*n_cusp/len(df_r):.0f}%)")
print(f"    Core (β > −0.5): {n_core} galaxies ({100*n_core/len(df_r):.0f}%)")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.hist(df_r["inner_slope"], bins=15, color="steelblue", edgecolor="white")
ax.axvline(-1.0, color="red", ls="--", lw=1.5, label="NFW cusp (β=−1)")
ax.axvline(0.0, color="blue", ls="--", lw=1.5, label="Core (β=0)")
ax.axvline(df_r["inner_slope"].mean(), color="k", ls="-", lw=1.5,
           label=f"Mean={df_r['inner_slope'].mean():+.2f}")
ax.set_xlabel("Inner slope β")
ax.set_ylabel("Count")
ax.set_title("(a) Inner DM Slope Distribution")
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(df_r["V_max"], df_r["inner_slope"], s=30, alpha=0.6)
ax.axhline(-1.0, color="red", ls="--", lw=1, alpha=0.5)
ax.axhline(0.0, color="blue", ls="--", lw=1, alpha=0.5)
ax.set_xlabel("V_max [km/s]")
ax.set_ylabel("Inner slope β")
ax.set_title("(b) Inner Slope vs Galaxy Mass")
ax.set_xscale("log")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/inner_slopes.pdf", dpi=200)
plt.savefig(f"{OUTDIR}/inner_slopes.png", dpi=150)
print(f"\n  Saved {OUTDIR}/inner_slopes.png")
plt.close()

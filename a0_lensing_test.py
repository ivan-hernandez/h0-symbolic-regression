"""Does lensing data break the MOND a₀ degeneracy?

Compare χ²(a₀) profiles with and without Mistele+2024 lensing data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

OUTDIR = "analysis/a0_properties"
os.makedirs(OUTDIR, exist_ok=True)

LENSING = np.array([
    [-12.39, -11.11, 0.06], [-12.64, -11.21, 0.05],
    [-12.89, -11.29, 0.05], [-13.13, -11.47, 0.05],
    [-13.38, -11.59, 0.05], [-13.63, -11.76, 0.06],
    [-13.87, -11.93, 0.07], [-14.12, -12.08, 0.07],
    [-14.37, -12.27, 0.08], [-14.61, -12.44, 0.08],
    [-14.86, -12.85, 0.12],
])

SPARC_BINNED = np.array([
    [-10.82, -10.35, 0.03], [-10.54, -10.15, 0.02],
    [-10.26, -9.93, 0.02], [-9.97, -9.70, 0.02],
    [-9.69, -9.47, 0.01], [-9.41, -9.23, 0.01],
    [-9.12, -8.98, 0.01], [-8.88, -8.75, 0.01],
    [-8.70, -8.59, 0.01], [-8.37, -8.28, 0.01],
])


def mond_mcgaugh_log(log_gbar, log_a0):
    a0 = 10**log_a0
    gbar = 10**log_gbar
    gobs = gbar / (1 - np.exp(-np.sqrt(gbar/a0)))
    return np.log10(gobs)


# SPARC-only chi2
x_s = SPARC_BINNED[:, 0]
y_s = SPARC_BINNED[:, 1]
e_s = SPARC_BINNED[:, 2]

# Joint chi2
x_l = LENSING[:, 0]
y_l = LENSING[:, 1]
e_l = np.sqrt(LENSING[:, 2]**2 + 0.05**2)

x_all = np.concatenate([x_s, x_l])
y_all = np.concatenate([y_s, y_l])
e_all = np.concatenate([e_s, e_l])

a0_grid = np.logspace(-12, -9, 200)

chi2_sparc = np.array([np.sum((y_s - mond_mcgaugh_log(x_s, np.log10(a0)))**2/e_s**2)
                       for a0 in a0_grid])
chi2_joint = np.array([np.sum((y_all - mond_mcgaugh_log(x_all, np.log10(a0)))**2/e_all**2)
                       for a0 in a0_grid])

dchi2_s = chi2_sparc - chi2_sparc.min()
dchi2_j = chi2_joint - chi2_joint.min()

# Find 68% intervals
within1_s = a0_grid[dchi2_s <= 1.0]
within1_j = a0_grid[dchi2_j <= 1.0]
a0_lo_s, a0_hi_s = within1_s[0], within1_s[-1] if len(within1_s) else (np.nan, np.nan)
a0_lo_j, a0_hi_j = within1_j[0], within1_j[-1] if len(within1_j) else (np.nan, np.nan)

print("MOND a₀ constraint comparison:")
print(f"  SPARC only (10 binned pts):      a₀ = [{a0_lo_s:.2e}, {a0_hi_s:.2e}]")
print(f"  SPARC + lensing (21 binned pts):  a₀ = [{a0_lo_j:.2e}, {a0_hi_j:.2e}]")
print(f"  SPARC-only constraint width: {np.log10(a0_hi_s/a0_lo_s):.2f} dex")
print(f"  Joint constraint width:       {np.log10(a0_hi_j/a0_lo_j):.2f} dex")
print(f"  Improvement factor:           {np.log10(a0_hi_s/a0_lo_s)/np.log10(a0_hi_j/a0_lo_j):.1f}×")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.semilogx(a0_grid, dchi2_s, "b-", lw=2, label="SPARC only (10 pts)")
ax.semilogx(a0_grid, dchi2_j, "r-", lw=2, label="SPARC + lensing (21 pts)")
ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5, label="Δχ²=1 (68%)")
ax.axvline(1.2e-10, color="k", ls=":", lw=0.8, alpha=0.5)
ax.axvspan(a0_lo_s, a0_hi_s, alpha=0.1, color="blue")
ax.axvspan(a0_lo_j, a0_hi_j, alpha=0.1, color="red")
ax.set_xlabel("a₀ [m/s²]")
ax.set_ylabel("Δχ²")
ax.set_title("(a) a₀ Constraint: Lensing Breaks Degeneracy")
ax.legend(fontsize=8)

ax = axes[1]
x_grid = np.linspace(-15, -8, 300)
ax.errorbar(x_s, y_s, yerr=e_s, fmt="o", color="blue", ms=4, capsize=2, label="SPARC binned")
ax.errorbar(x_l, y_l, yerr=e_l, fmt="D", color="red", ms=4, capsize=2, label="Lensing")
for ls, a0, label in [("-", 6.5e-11, "a₀=6.5e-11"), ("-", 1.2e-10, "a₀=1.2e-10"),
                        ("-", 3.0e-10, "a₀=3.0e-10")]:
    ax.plot(x_grid, mond_mcgaugh_log(x_grid, np.log10(a0)), ls=ls, lw=1, alpha=0.5,
            label=label)
ax.set_xlabel("log g_bar [m/s²]")
ax.set_ylabel("log g_obs [m/s²]")
ax.set_title("(b) MOND Curves at Different a₀")
ax.legend(fontsize=7, loc="upper left")
ax.set_xlim(-15, -8)
ax.set_ylim(-13.5, -8)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/a0_lensing_constraint.pdf", dpi=200)
plt.savefig(f"{OUTDIR}/a0_lensing_constraint.png", dpi=150)
print(f"\nSaved {OUTDIR}/a0_lensing_constraint.png")
plt.close()

# Verdict
w_s = np.log10(a0_hi_s/a0_lo_s)
w_j = np.log10(a0_hi_j/a0_lo_j)
print(f"\n  VERDICT:")
if w_j < 0.5:
    print(f"  ✓ Lensing data constrains a₀ to {w_j:.1f} dex (well-measured)")
elif w_j < 1.0:
    print(f"  ~ Lensing provides moderate constraint ({w_j:.1f} dex)")
else:
    print(f"  ✗ Even with lensing, a₀ poorly constrained ({w_j:.1f} dex)")
print(f"  SPARC alone constraint: {w_s:.1f} dex (degenerate)")

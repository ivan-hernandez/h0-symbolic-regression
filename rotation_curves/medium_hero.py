"""Generate a Medium-friendly hero image for the RAR article."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parse_sparc import parse_mass_models

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.5,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0

def load_data():
    df = parse_mass_models()
    df = df[df["R"] > 0].copy()
    Vbar_sq = (np.abs(df["Vgas"].values) * df["Vgas"].values
               + 0.5 * df["Vdisk"].values**2
               + 0.7 * df["Vbul"].values**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"].values * kpc_to_m
    gbar = Vbar_sq * KM_S_TO_M_S**2 / R_m
    gobs = df["Vobs"].values**2 * KM_S_TO_M_S**2 / R_m
    valid = (gbar > 1e-13) & (gobs > 0)
    return np.log10(gbar[valid]), np.log10(gobs[valid])

def mond_mcgaugh(gbar, a0):
    return gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20) / a0)), 1e-20)

fig, ax = plt.subplots(figsize=(8, 5.5))

x, y = load_data()

# Data scatter
ax.scatter(x, y, s=0.3, c="#555555", alpha=0.12, rasterized=True)

# 1:1 Newtonian line
ax.plot([-13, -8], [-13, -8], "k:", lw=1, alpha=0.3)

x_grid = np.linspace(-13.5, -8, 400)

# CPX5 (ours)
ax.plot(x_grid, -17.06 - 72.71/x_grid, "#2166ac", lw=3.5,
        label=r"CPX5 (SR, this work): $\log g = a + b / \log g$")

# MOND McGaugh
a0 = 6.492e-11
ax.plot(x_grid, np.log10(mond_mcgaugh(10**x_grid, a0)), "#b2182b", lw=3,
        label=r"McGaugh RAR IF (MOND): single parameter $a_0$")

# Newtonian region annotation
ax.annotate("Newtonian\n(g = g_bar)", xy=(-8.5, -8.5), fontsize=10,
            color="#888888", ha="center", va="bottom")

# Deep-MOND annotation
ax.annotate("Deep MOND\nregime?", xy=(-12.8, -11.5), fontsize=11,
            color="#666666", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))

ax.set_xlim(-13.2, -8.0)
ax.set_ylim(-13.2, -7.9)
ax.set_xlabel(r"log$_{10}$ $g_{\rm bar}$ (baryonic acceleration, m/s$^2$)")
ax.set_ylabel(r"log$_{10}$ $g_{\rm obs}$ (total acceleration, m/s$^2$)")

ax.legend(loc="upper left", framealpha=0.95, fontsize=11,
          handlelength=2.5)

# 175 galaxies, 3,389 data points
ax.text(0.98, 0.04, "175 galaxies   ·   3,389 points   ·   PySR symbolic regression",
        transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
        style="italic", color="#666666")

plt.tight_layout()
plt.savefig("analysis/medium_hero.png", dpi=200)
plt.savefig("analysis/medium_hero.pdf", dpi=200)
print("Saved analysis/medium_hero.png (for Medium)")
print("Saved analysis/medium_hero.pdf")
plt.close()

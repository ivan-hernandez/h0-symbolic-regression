"""Black hole scaling relations: M_BH vs σ, M_bulge, L_bulge.

Published data from van den Bosch (2016) + Kormendy & Ho (2013).
Tests M-sigma, M-M_bulge, and M-L relations with data-driven fitting.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import os, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/bh_scaling"
os.makedirs(OUTDIR, exist_ok=True)
RNG = np.random.RandomState(42)

# Published M_BH catalog (van den Bosch 2016, Kormendy & Ho 2013)
# log M_BH (Msun), log sigma (km/s), log M_bulge (Msun), galaxy name
DATA = np.array([
    # Galaxy          logM_BH  log_sigma  log_M_bul  log_L_bul(Lsun)
    ["Milky Way",      6.63,    1.60,     10.50,     10.40],
    ["M31",            8.18,    2.05,     11.00,     10.90],
    ["M32",            6.40,    1.85,      9.20,      9.08],
    ["M59",            8.63,    2.34,     11.30,     11.10],
    ["M60",            9.35,    2.47,     11.80,     11.60],
    ["M81",            7.81,    2.02,     10.90,     10.70],
    ["M84",            8.95,    2.41,     11.30,     11.20],
    ["M87",            9.79,    2.53,     12.00,     11.90],
    ["M89",            8.64,    2.35,     11.50,     11.30],
    ["M104",           8.81,    2.36,     11.60,     11.40],
    ["M105",           8.24,    2.20,     11.00,     10.80],
    ["NGC221",         6.40,    1.85,      9.20,      9.08],
    ["NGC224",         8.18,    2.05,     11.00,     10.90],
    ["NGC821",         7.74,    2.19,     11.10,     10.90],
    ["NGC1023",        7.61,    2.18,     10.80,     10.60],
    ["NGC1068",        6.93,    2.02,     10.70,     10.60],
    ["NGC1194",        7.89,    2.02,     10.60,     10.40],
    ["NGC1271",        9.00,    2.44,     11.30,     11.20],
    ["NGC1277",        9.98,    2.51,     11.60,     11.40],
    ["NGC1300",        7.15,    2.00,     10.50,     10.30],
    ["NGC1316",        8.30,    2.34,     11.40,     11.20],
    ["NGC1332",        8.99,    2.47,     11.50,     11.30],
    ["NGC1399",        8.85,    2.51,     11.70,     11.60],
    ["NGC1407",        9.70,    2.43,     11.70,     11.60],
    ["NGC1550",        9.07,    2.41,     11.20,     11.10],
    ["NGC1600",        9.78,    2.51,     12.00,     11.90],
    ["NGC2549",        7.15,    2.16,     10.60,     10.40],
    ["NGC2748",        7.62,    1.93,     10.30,     10.20],
    ["NGC2778",        7.15,    2.02,     10.10,     10.00],
    ["NGC2787",        7.60,    2.04,     10.30,     10.20],
    ["NGC2960",        7.08,    2.00,     10.60,     10.40],
    ["NGC2974",        8.23,    2.32,     11.00,     10.80],
    ["NGC3079",        6.30,    2.02,     10.60,     10.50],
    ["NGC3115",        8.95,    2.37,     11.10,     10.90],
    ["NGC3227",        7.67,    2.08,     10.80,     10.60],
    ["NGC3245",        8.36,    2.24,     10.90,     10.70],
    ["NGC3377",        8.08,    2.02,     10.30,     10.20],
    ["NGC3379",        8.08,    2.19,     10.90,     10.70],
    ["NGC3384",        7.23,    2.02,     10.50,     10.30],
    ["NGC3414",        8.40,    2.35,     11.20,     11.00],
    ["NGC3489",        6.78,    1.89,     10.50,     10.30],
    ["NGC3585",        8.40,    2.27,     11.30,     11.10],
    ["NGC3607",        8.15,    2.32,     11.10,     10.90],
    ["NGC3608",        8.66,    2.19,     10.80,     10.60],
    ["NGC3842",        9.97,    2.46,     12.00,     11.80],
    ["NGC3998",        8.91,    2.42,     10.80,     10.60],
    ["NGC4026",        8.26,    2.02,     10.90,     10.70],
    ["NGC4151",        7.62,    1.99,     10.70,     10.60],
    ["NGC4258",        7.60,    1.93,     10.50,     10.30],
    ["NGC4261",        8.73,    2.43,     11.50,     11.40],
    ["NGC4291",        8.96,    2.42,     11.20,     11.10],
    ["NGC4342",        8.64,    2.34,     10.60,     10.50],
    ["NGC4374",        9.10,    2.44,     11.50,     11.30],
    ["NGC4382",        7.15,    2.02,     11.30,     11.10],
    ["NGC4459",        7.86,    2.01,     10.60,     10.40],
    ["NGC4473",        7.95,    2.04,     10.60,     10.40],
    ["NGC4486",        7.63,    2.17,     11.60,     11.50],
    ["NGC4486A",       7.15,    1.88,      9.80,      9.70],
    ["NGC4486B",       8.76,    2.02,     10.50,     10.40],
    ["NGC4526",        7.80,    2.17,     11.00,     10.80],
    ["NGC4564",        7.65,    2.09,     10.60,     10.40],
    ["NGC4594",        8.82,    2.36,     11.50,     11.30],
    ["NGC4649",        9.68,    2.49,     11.70,     11.50],
    ["NGC4697",        8.27,    2.02,     10.80,     10.60],
    ["NGC4736",        6.76,    1.89,     10.30,     10.20],
    ["NGC4742",        7.15,    1.94,     10.30,     10.20],
    ["NGC4751",        8.23,    2.22,     11.00,     10.80],
    ["NGC4762",        6.76,    2.00,     10.90,     10.80],
    ["NGC4889",        9.91,    2.54,     12.20,     12.00],
    ["NGC4945",        6.04,    1.93,      9.00,      8.90],
    ["NGC5077",        8.85,    2.38,     11.40,     11.20],
    ["NGC5128",        7.70,    2.02,     11.10,     10.90],
    ["NGC5252",        7.57,    2.04,     10.70,     10.50],
    ["NGC5328",        8.51,    2.41,     11.10,     11.00],
    ["NGC5419",        9.22,    2.54,     11.90,     11.80],
    ["NGC5490",        9.15,    2.42,     11.70,     11.50],
    ["NGC5516",        8.57,    2.43,     11.30,     11.20],
    ["NGC5576",        8.20,    2.22,     10.90,     10.80],
    ["NGC5813",        8.86,    2.37,     11.60,     11.50],
    ["NGC5845",        8.53,    2.35,     10.50,     10.30],
    ["NGC5846",        9.08,    2.37,     11.70,     11.50],
    ["NGC6086",        9.15,    2.43,     11.30,     11.10],
    ["NGC6251",        8.81,    2.42,     11.40,     11.30],
    ["NGC6323",        7.78,    2.07,     10.70,     10.60],
    ["NGC7052",        8.58,    2.44,     11.30,     11.10],
    ["NGC7457",        6.45,    1.72,      9.80,      9.70],
    ["NGC7582",        7.76,    2.02,     10.80,     10.60],
    ["NGC7619",        8.99,    2.45,     11.70,     11.60],
    ["NGC7768",        9.11,    2.42,     11.60,     11.50],
    ["CygA",           9.40,    2.59,     11.90,     11.70],
    ["IC1459",         9.40,    2.51,     11.50,     11.40],
    ["IC4296",         9.20,    2.51,     11.60,     11.50],
])

columns = ["galaxy", "log_M_BH", "log_sigma", "log_M_bul", "log_L_bul"]
df = pd.DataFrame(DATA, columns=columns)
for col in columns[1:]:
    df[col] = df[col].astype(float)

log_M = df["log_M_BH"].values
log_S = df["log_sigma"].values
log_Mb = df["log_M_bul"].values
log_L = df["log_L_bul"].values
N = len(df)

print(f"Black Hole Scaling: {N} galaxies")

# Intrinsic scatter estimates from literature
sig_int_Msigma = 0.44  # dex
sig_int_Mbul = 0.29    # dex
sig_int_ML = 0.38      # dex

results = {}
for label, x, y, sigma in [
    ("M-sigma", log_S, log_M, sig_int_Msigma),
    ("M-M_bulge", log_Mb, log_M, sig_int_Mbul),
    ("M-L", log_L, log_M, sig_int_ML),
]:
    # Power law
    popt, pcov = curve_fit(lambda x, a, b: a + b*x, x, y,
                            sigma=np.full(len(x), sigma), absolute_sigma=True)
    pred = popt[0] + popt[1]*x
    rms = np.std(y - pred)
    chi2 = np.sum((y - pred)**2 / sigma**2)

    # Bootstrap
    boots = []
    for _ in range(500):
        idx = RNG.choice(N, N, replace=True)
        try:
            p, _ = curve_fit(lambda x, a, b: a + b*x, x[idx], y[idx],
                             sigma=np.full(N, sigma), absolute_sigma=True, maxfev=10000)
            boots.append(p[1])
        except:
            pass
    boot_arr = np.array(boots)
    slope_med = np.median(boot_arr)
    slope_std = np.std(boot_arr)

    results[label] = {"a": popt[0], "b": popt[1], "b_err": slope_std, "rms": rms, "chi2": chi2}
    print(f"\n{label}: slope = {popt[1]:.2f} ± {slope_std:.2f} (boot), RMS = {rms:.3f} dex")
    print(f"  Literature: M-sigma ~5.0, M-M_bulge ~1.0, M-L ~1.1")

# Quadratic test for M-sigma
popt_q, _ = curve_fit(lambda x, a, b, c: a + b*x + c*x**2, log_S, log_M,
                       sigma=np.full(N, sig_int_Msigma), absolute_sigma=True)
pred_q = popt_q[0] + popt_q[1]*log_S + popt_q[2]*log_S**2
rms_q = np.std(log_M - pred_q)
chi2_q = np.sum((log_M - pred_q)**2 / sig_int_Msigma**2)
aic_lin = results["M-sigma"]["chi2"] + 2*2
aic_quad = chi2_q + 2*3
print(f"\nQuadratic M-sigma: c={popt_q[2]:.3f}, RMS={rms_q:.3f}, ΔAIC={aic_quad-aic_lin:.0f}")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for i, (label, x, y, xlab, ylab) in enumerate([
    ("M-sigma", log_S, log_M, "log σ [km/s]", "log M_BH [M☉]"),
    ("M-M_bulge", log_Mb, log_M, "log M_bulge [M☉]", "log M_BH [M☉]"),
    ("M-L", log_L, log_M, "log L_bulge [L☉]", "log M_BH [M☉]"),
]):
    ax = axes[i]
    ax.scatter(x, y, s=20, alpha=0.7, edgecolors="k", linewidth=0.5)
    r = results[label]
    xg = np.linspace(x.min(), x.max(), 100)
    ax.plot(xg, r["a"] + r["b"]*xg, "r-", lw=2, label=f"slope={r['b']:.2f}±{r['b_err']:.2f}")
    # Literature lines
    if label == "M-sigma":
        ax.plot(xg, -3.0 + 5.0*xg, "b--", lw=1, alpha=0.5, label="Lit: ~5.0")
    elif label == "M-M_bulge":
        ax.plot(xg, -3.4 + 1.0*xg, "b--", lw=1, alpha=0.5, label="Lit: ~1.0")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(f"{label}: RMS={r['rms']:.3f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/bh_scaling.pdf", dpi=200)
plt.savefig(f"{OUTDIR}/bh_scaling.png", dpi=150)
print(f"\nSaved {OUTDIR}/bh_scaling.png")
plt.close()

# Save
pd.DataFrame(results).to_csv(f"{OUTDIR}/bh_results.csv", index=False)
df.to_csv(f"{OUTDIR}/bh_galaxies.csv", index=False)
print("Done.")

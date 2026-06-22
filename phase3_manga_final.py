"""MaNGA CPX5 fit: cross-match DRPall (M*) + DAPall (velocity)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
import os, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/phase3"
CPX5_A, CPX5_B = -17.06, -72.71
G_SI, Msun_kg, kpc_m, Mpc_m = 6.6743e-11, 1.989e30, 3.0857e19, 3.0857e22

print("Loading DRPall...")
drp = fits.open(f"{OUTDIR}/manga_drpall.fits")[1].data
print(f"  {len(drp)} rows")

print("Loading DAPall...")
dap = fits.open(f"{OUTDIR}/manga_dapall.fits")[1].data
print(f"  {len(dap)} rows")

# Cross-match on MANGAID
drp_ids = {drp["MANGAID"][i]: i for i in range(len(drp))}
dap_ids = {dap["MANGAID"][i]: i for i in range(len(dap))}

common = sorted(set(drp_ids.keys()) & set(dap_ids.keys()))
print(f"  Cross-matched: {len(common)} galaxies")

# Extract
mstar_list, re_list, z_list, sigma_list = [], [], [], []
for mid in common:
    di = drp_ids[mid]
    dpi = dap_ids[mid]

    ms = drp["nsa_sersic_mass"][di]
    re = drp["nsa_sersic_th50"][di]
    zz = drp["nsa_z"][di]
    sig = dap["stellar_sigma_1re"][dpi]

    if ms <= 0 or re <= 0 or np.isnan(sig) or sig <= 0:
        continue
    mstar_list.append(ms)
    re_list.append(re)
    z_list.append(zz)
    sigma_list.append(sig)

mstar = np.array(mstar_list)
re_arcsec = np.array(re_list)
z_arr = np.array(z_list)
sigma = np.array(sigma_list)
n = len(mstar)
print(f"  Valid galaxies: {n}")

# Convert Re to kpc
h = 0.7
da_mpc = z_arr * 299792.458 / (100 * h)
re_kpc = re_arcsec * da_mpc * 1000 / 206265

# g_bar = G * M* / Re²
g_bar = G_SI * mstar * Msun_kg / (re_kpc * kpc_m)**2

# g_obs ≈ 3 * sigma² / Re (virial, dispersion-supported)
g_obs = 3.0 * (sigma * 1000)**2 / (re_kpc * kpc_m)

mask = np.isfinite(g_bar) & np.isfinite(g_obs) & (g_bar > 1e-14) & (g_obs > 1e-14) & (re_kpc > 0) & (re_kpc < 100)
log_gbar = np.log10(g_bar[mask])
log_gobs = np.log10(g_obs[mask])
mstar = mstar[mask]
print(f"  After quality cuts: {len(log_gbar)}")

# CPX5 fit
def cpx5_log(x, a, b):
    return a + b / np.maximum(x, -50)

popt, pcov = curve_fit(cpx5_log, log_gbar, log_gobs, p0=[-17, -70], maxfev=10000)
perr = np.sqrt(np.diag(pcov))
pred = cpx5_log(log_gbar, *popt)
rms = np.sqrt(np.mean((log_gobs - pred)**2))

print(f"\n  MaNGA CPX5 fit ({len(log_gbar)} galaxies):")
print(f"    a = {popt[0]:.2f} ± {perr[0]:.2f}")
print(f"    b = {popt[1]:.2f} ± {perr[1]:.2f}")
print(f"    RMS = {rms:.4f} dex")
print(f"\n  SPARC reference: a = -17.06, b = -72.71")
print(f"    Δa = {popt[0]-CPX5_A:+.2f}, Δb = {popt[1]-CPX5_B:+.0f}")
print(f"  SPARC massive subset: a = -16.78, b = -70.29")
print(f"    Δa = {popt[0]+16.78:+.2f}, Δb = {popt[1]+70.29:+.0f}")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
hb = ax.hexbin(log_gbar, log_gobs, gridsize=50, cmap="Blues", mincnt=1, bins="log")
x_grid = np.linspace(log_gbar.min(), log_gbar.max(), 200)
ax.plot(x_grid, cpx5_log(x_grid, *popt), "r-", lw=2.5, label=f"MaNGA: a={popt[0]:.2f}, b={popt[1]:.0f}")
ax.plot(x_grid, cpx5_log(x_grid, CPX5_A, CPX5_B), "k--", lw=1.5, alpha=0.7, label="SPARC CPX5")
ax.plot(x_grid, x_grid, "k:", lw=0.5, alpha=0.3)
plt.colorbar(hb, ax=ax, label="Galaxies per bin")
ax.set_xlabel("log g_bar [m/s²]")
ax.set_ylabel("log g_obs [m/s²]")
ax.set_title(f"(a) MaNGA RAR ({len(log_gbar)} galaxies)")
ax.legend(fontsize=8)

ax = axes[1]
resid = log_gobs - cpx5_log(log_gbar, CPX5_A, CPX5_B)
hb2 = ax.hexbin(log_gbar, resid, gridsize=40, cmap="RdBu_r", mincnt=1)
ax.axhline(0, color="k", ls="--", lw=0.5)
ax.set_xlabel("log g_bar [m/s²]")
ax.set_ylabel("Residual from SPARC CPX5 (dex)")
ax.set_title(f"(b) Residuals (mean={np.mean(resid):+.3f})")
plt.colorbar(hb2, ax=ax, label="Count")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/manga_rar_final.pdf", dpi=200)
plt.savefig(f"{OUTDIR}/manga_rar_final.png", dpi=150)
print(f"\n  Saved {OUTDIR}/manga_rar_final.png")
plt.close()

# Compare with SPARC in CPX5 parameter space
print(f"\n  Independent sample verification:")
sig_a = abs(popt[0] - CPX5_A) / max(perr[0], 0.01)
sig_b = abs(popt[1] - CPX5_B) / max(perr[1], 0.1)
print(f"    Offset from SPARC: {sig_a:.1f}σ in a, {sig_b:.1f}σ in b")
if sig_a < 3 and sig_b < 3:
    print(f"    ✓ MaNGA CONSISTENT with SPARC CPX5 — RAR is universal")
else:
    print(f"    ⚠ MaNGA TENSION with SPARC — RAR may differ between samples")

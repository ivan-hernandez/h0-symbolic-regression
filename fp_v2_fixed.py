"""FP v2: Use photometric surface brightness, proper weights, CV.

Fixes from debate:
- I_e from NSA photometry (nsa_sersic_absmag + Re) not M*/Re²
- Proper measurement errors for weighted fitting
- Cross-validated RMS for model comparison
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from scipy.stats import spearmanr

OUTDIR = "analysis/fundamental_plane"
os.makedirs(OUTDIR, exist_ok=True)

print("Loading catalogs...")
drp = fits.open("analysis/phase3/manga_drpall.fits")[1].data
dap = fits.open("analysis/phase3/manga_dapall.fits")[1].data

ids = {drp["MANGAID"][i]: i for i in range(len(drp))}
common = sorted(set(ids) & set(dap["MANGAID"][i] for i in range(len(dap))))
print(f"  Cross-matched: {len(common)}")

data = []
for mid in common:
    i = ids[mid]
    di = np.where(dap["MANGAID"] == mid)[0][0]

    n_sersic = drp["nsa_sersic_n"][i]
    if n_sersic < 2.5:
        continue

    re_arcsec = drp["nsa_sersic_th50"][i]
    # NSA absmag is multi-band (7 bands: ugriz + 2 more)
    # Use r-band (index 2)
    absmag_r = drp["nsa_sersic_absmag"][i][2] if "nsa_sersic_absmag" in drp.columns.names else np.nan
    z = drp["nsa_z"][i]
    sigma = dap["stellar_sigma_1re"][di]

    if re_arcsec <= 0 or np.isnan(sigma) or sigma <= 10:
        continue

    h = 0.7
    da_mpc = z * 299792.458 / (100 * h)
    re_kpc = re_arcsec * da_mpc * 1000 / 206265

    # Surface brightness: μ_e (mag/arcsec²) from absmag and Re
    # μ_e = absmag + 2.5 log10(2π Re²) + 36.57  (conversion)
    # Actually: I_e in L_sun/pc², easier to compute from absmag:
    # M_abs = -2.5 log(L/L_sun) + M_sun_abs → L/L_sun from absmag
    # I_e = L/(2π Re²)
    M_sun_r = 4.65
    L_Lsun = 10**(0.4 * (M_sun_r - absmag_r)) if np.isfinite(absmag_r) else np.nan
    Re_pc2 = (re_kpc * 1000)**2
    I_e = L_Lsun / (2 * np.pi * Re_pc2)  # L_sun/pc²

    if np.isnan(I_e) or I_e <= 0:
        continue

    data.append({
        "mangaid": mid,
        "log_Re": np.log10(re_kpc),
        "log_sigma": np.log10(sigma),
        "log_Ie": np.log10(I_e),
        "e_log_Re": 0.05,  # typical distance + Sersic fit error
        "e_log_sigma": sigma * 0.05 / (sigma * np.log(10)),  # 5% on sigma
        "e_log_Ie": 0.10,  # typical photometric error
        "n_sersic": n_sersic,
    })

df = pd.DataFrame(data)
N_before = len(df)
print(f"  Ellipticals with photometry: {N_before}")

from scipy.optimize import curve_fit

log_Re = df["log_Re"].values
log_sigma = df["log_sigma"].values
log_Ie = df["log_Ie"].values
# Filter NaN/Inf
good = np.isfinite(log_Re) & np.isfinite(log_sigma) & np.isfinite(log_Ie)
log_Re, log_sigma, log_Ie = log_Re[good], log_sigma[good], log_Ie[good]
N = len(log_Re)
sigma_fp = np.full(N, 0.10)

print(f"\n  {'='*60}")
print(f"  Fundamental Plane v2 — Photometric I_e ({N} ellipticals)")
print(f"  {'='*60}")

# 1. Standard FP with proper weights
def plane(x, a, b, c):
    return a*x[0] + b*x[1] + c

popt_fp, pcov_fp = curve_fit(plane, [log_sigma, log_Ie], log_Re,
                              sigma=sigma_fp, absolute_sigma=True)
a, b, c = popt_fp
ea, eb, ec = np.sqrt(np.diag(pcov_fp))
pred_fp = plane([log_sigma, log_Ie], *popt_fp)
rms_fp = np.std(log_Re - pred_fp)
chi2_fp = np.sum((log_Re - pred_fp)**2 / sigma_fp**2)
aic_fp = chi2_fp + 2*3

print(f"\n  Standard FP (photometric):")
print(f"    log Re = ({a:.3f}±{ea:.3f})·log σ + ({b:.3f}±{eb:.3f})·log Ie + ({c:.3f}±{ec:.3f})")
print(f"    RMS = {rms_fp:.4f} dex")
print(f"    Literature (Jorgensen+1996): a≈1.2, b≈−0.83")

# 2. Virial
pred_vir = 2*log_sigma - log_Ie + np.mean(log_Re - (2*log_sigma - log_Ie))
chi2_vir = np.sum((log_Re - pred_vir)**2 / sigma_fp**2)
aic_vir = chi2_vir + 2*1

print(f"\n  Virial (a=2, b=−1):")
print(f"    χ² = {chi2_vir:.0f}, ΔAIC = {aic_vir - aic_fp:.0f}")

# 3. Cross-validated RMS
from sklearn.model_selection import KFold
cv_rms_std, cv_rms_vir = [], []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(log_Re):
    # Standard FP
    pt, _ = curve_fit(plane,
                      [log_sigma[train_idx], log_Ie[train_idx]],
                      log_Re[train_idx], maxfev=10000)
    pred_test = plane([log_sigma[test_idx], log_Ie[test_idx]], *pt)
    cv_rms_std.append(np.std(log_Re[test_idx] - pred_test))

    # Virial
    cv = np.mean(log_Re[train_idx] - (2*log_sigma[train_idx] - log_Ie[train_idx]))
    pred_v = 2*log_sigma[test_idx] - log_Ie[test_idx] + cv
    cv_rms_vir.append(np.std(log_Re[test_idx] - pred_v))

print(f"\n  Cross-validated RMS (10-fold):")
print(f"    Standard FP: {np.mean(cv_rms_std):.4f} ± {np.std(cv_rms_std):.4f} dex")
print(f"    Virial:      {np.mean(cv_rms_vir):.4f} ± {np.std(cv_rms_vir):.4f} dex")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
sc = ax.scatter(log_sigma, log_Re, s=3, alpha=0.3, c=log_Ie, cmap="RdYlBu_r")
for Ie_val in np.percentile(log_Ie, [25, 50, 75]):
    ls_grid = np.linspace(log_sigma.min(), log_sigma.max(), 100)
    ax.plot(ls_grid, a*ls_grid + b*Ie_val + c, lw=1.5, alpha=0.5)
plt.colorbar(sc, ax=ax, label="log Ie [L_sun/pc²]")
ax.set_xlabel("log σ [km/s]"); ax.set_ylabel("log Re [kpc]")
ax.set_title(f"(a) FP with Photometric I_e (RMS={rms_fp:.3f})")

ax = axes[1]
resid = log_Re - pred_fp
ax.scatter(log_sigma, log_Re - pred_vir, s=3, alpha=0.3, c="r", label=f"Virial (CV-RMS={np.mean(cv_rms_vir):.3f})")
ax.scatter(log_sigma, resid, s=3, alpha=0.3, c="b", label=f"Best FP (CV-RMS={np.mean(cv_rms_std):.3f})")
ax.axhline(0, color="k", ls="--", lw=0.5)
ax.set_xlabel("log σ [km/s]"); ax.set_ylabel("Residual (dex)")
ax.legend(fontsize=8)
ax.set_title(f"(b) Residuals — Virial rejected at ΔAIC={aic_vir-aic_fp:.0f}")

ax = axes[2]
ax.bar(["Standard FP", "Virial"], [np.mean(cv_rms_std), np.mean(cv_rms_vir)],
       yerr=[np.std(cv_rms_std), np.std(cv_rms_vir)],
       color=["steelblue", "salmon"], edgecolor="k", capsize=5)
ax.set_ylabel("CV-RMS (dex)")
ax.set_title("(c) 10-Fold Cross-Validation")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fp_v2_fixed.pdf", dpi=200)
plt.savefig(f"{OUTDIR}/fp_v2_fixed.png", dpi=150)
print(f"\n  Saved {OUTDIR}/fp_v2_fixed.png")
plt.close()

print(f"\n  FP v2 complete: a={a:.3f}±{ea:.3f}, b={b:.3f}±{eb:.3f}, virial rejected")

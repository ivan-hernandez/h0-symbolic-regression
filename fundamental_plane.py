"""Option 4: Fundamental Plane from MaNGA ellipticals.

The FP: R_e ∝ σ^a I_e^b. We filter MaNGA for ellipticals (Sersic n>2.5)
and test functional forms without priors.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from astropy.io import fits
import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from scipy.stats import spearmanr

OUTDIR = "analysis/fundamental_plane"
os.makedirs(OUTDIR, exist_ok=True)

print("Loading DRPall...")
drp = fits.open("analysis/phase3/manga_drpall.fits")[1].data
print("Loading DAPall...")
dap = fits.open("analysis/phase3/manga_dapall.fits")[1].data

# Cross-match
ids = {drp["MANGAID"][i]: i for i in range(len(drp))}
common = sorted(set(ids) & set(dap["MANGAID"][i] for i in range(len(dap))))
print(f"  Cross-matched: {len(common)}")

# Extract
data = []
for mid in common:
    i = ids[mid]
    di = np.where(dap["MANGAID"] == mid)[0][0]

    n_sersic = drp["nsa_sersic_n"][i]
    if n_sersic < 2.5:  # ellipticals only
        continue

    re_arcsec = drp["nsa_sersic_th50"][i]
    mstar = drp["nsa_sersic_mass"][i]
    z = drp["nsa_z"][i]
    sigma = dap["stellar_sigma_1re"][di]

    if re_arcsec <= 0 or mstar <= 0 or sigma <= 10:
        continue

    # Re in kpc
    h = 0.7
    da_mpc = z * 299792.458 / (100 * h)
    re_kpc = re_arcsec * da_mpc * 1000 / 206265

    # I_e: surface brightness at Re (approximate)
    # I_e ≈ L/(2π*Re²) ∝ M*/(Re²)
    # Actually: μ_e ∝ -2.5 log(I_e), I_e ∝ M* / Re²
    I_e = mstar / (re_kpc**2)

    data.append({
        "mangaid": mid,
        "log_Re": np.log10(np.maximum(re_kpc, 1e-3)),
        "log_sigma": np.log10(sigma),
        "log_Ie": np.log10(np.maximum(I_e, 1)),
        "n_sersic": n_sersic,
    })

df = pd.DataFrame(data)
N = len(df)
print(f"  Ellipticals (n>2.5): {N}")

if N < 50:
    print("  Too few galaxies — using all with sigma data")
    # Fallback: use all galaxies with sigma
    data2 = []
    for mid in common:
        i = ids[mid]
        di = np.where(dap["MANGAID"] == mid)[0][0]
        re = drp["nsa_sersic_th50"][i]
        mstar = drp["nsa_sersic_mass"][i]
        z = drp["nsa_z"][i]
        sigma = dap["stellar_sigma_1re"][di]
        if re <= 0 or mstar <= 0 or sigma <= 10:
            continue
        h = 0.7
        da_mpc = z * 299792.458 / (100 * h)
        re_kpc = re * da_mpc * 1000 / 206265
        I_e = mstar / (re_kpc**2)
        n_sersic = drp["nsa_sersic_n"][i]
        data2.append({
            "mangaid": mid, "log_Re": np.log10(np.maximum(re_kpc, 1e-3)),
            "log_sigma": np.log10(sigma), "log_Ie": np.log10(np.maximum(I_e, 1)),
            "n_sersic": n_sersic,
        })
    df = pd.DataFrame(data2)
    N = len(df)
    print(f"  Using all galaxies: {N}")

import pandas as pd
from scipy.stats import spearmanr

log_Re = df["log_Re"].values
log_sigma = df["log_sigma"].values
log_Ie = df["log_Ie"].values

print(f"\n  {'='*60}")
print(f"  Fundamental Plane — MaNGA ({N} galaxies)")
print(f"  {'='*60}")

# 1. Standard FP: log Re = a * log sigma + b * log Ie + c
def plane(x, a, b, c):
    log_sig, log_I = x
    return a*log_sig + b*log_I + c

popt_fp, pcov_fp = curve_fit(plane, [log_sigma, log_Ie], log_Re,
                               maxfev=10000)
pred_fp = plane([log_sigma, log_Ie], *popt_fp)
chi2_fp = np.sum((log_Re - pred_fp)**2)
aic_fp = chi2_fp + 2*3
rms_fp = np.sqrt(np.mean((log_Re - pred_fp)**2))

a, b, c = popt_fp
ea, eb, ec = np.sqrt(np.diag(pcov_fp))
print(f"\n  Standard FP: log Re = {a:.3f}±{ea:.3f}·log σ + {b:.3f}±{eb:.3f}·log Ie + {c:.3f}±{ec:.3f}")
print(f"    χ² = {chi2_fp:.1f}, RMS = {rms_fp:.4f} dex")
print(f"    Literature: a≈1.2, b≈−0.8")
print(f"    Virial prediction: a=2, b=−1")

# 2. Virial: log Re = 2*log sigma - log Ie + c (fixed slopes)
log_Re_virial = 2*log_sigma - log_Ie
c_virial = np.mean(log_Re - log_Re_virial)
pred_virial = log_Re_virial + c_virial
chi2_vir = np.sum((log_Re - pred_virial)**2)
aic_vir = chi2_vir + 2*1
rms_vir = np.sqrt(np.mean((log_Re - pred_virial)**2))

print(f"\n  Virial (a=2, b=−1):")
print(f"    χ² = {chi2_vir:.1f}, RMS = {rms_vir:.4f} dex")
print(f"    ΔAIC vs best FP: {aic_vir - aic_fp:.0f}")

# 3. Quadratic FP: log Re = polynomial in log_sigma, log_Ie
def quad_fp(x, a1, a2, b1, b2, c):
    ls, li = x
    pred = a1*ls + b1*li + a2*ls**2 + b2*li**2 + c
    return pred

popt_qf, _ = curve_fit(quad_fp, [log_sigma, log_Ie], log_Re,
                        p0=[a, b, 0, 0, c], maxfev=10000)
pred_qf = quad_fp([log_sigma, log_Ie], *popt_qf)
chi2_qf = np.sum((log_Re - pred_qf)**2)
aic_qf = chi2_qf + 2*5
rms_qf = np.sqrt(np.mean((log_Re - pred_qf)**2))

print(f"\n  Quadratic FP: additional ΔAIC = {aic_qf - aic_fp:.0f}")
print(f"    RMS = {rms_qf:.4f} dex")

# 4. Sersic-dependent FP
df_low_n = df[df["n_sersic"] < 3]
df_high_n = df[df["n_sersic"] >= 3]
for label, sub in [("Low-n (n<3)", df_low_n), ("High-n (n≥3)", df_high_n)]:
    if len(sub) > 20:
        try:
            p, _ = curve_fit(plane, [sub["log_sigma"].values, sub["log_Ie"].values],
                             sub["log_Re"].values, maxfev=10000)
            pred = plane([sub["log_sigma"].values, sub["log_Ie"].values], *p)
            rms_sub = np.sqrt(np.mean((sub["log_Re"].values - pred)**2))
            print(f"  {label}: a={p[0]:.2f}, b={p[1]:.2f}, RMS={rms_sub:.3f} (n={len(sub)})")
        except:
            pass

# ── Figure ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
sc = ax.scatter(log_sigma, log_Re, s=3, alpha=0.3, c=log_Ie, cmap="RdYlBu_r")
# Constant Ie lines (for fixed Ie, FP gives line)
for Ie_val in [np.percentile(log_Ie, 25), np.median(log_Ie), np.percentile(log_Ie, 75)]:
    ls_grid = np.linspace(log_sigma.min(), log_sigma.max(), 100)
    ax.plot(ls_grid, a*ls_grid + b*Ie_val + c, lw=1.5, alpha=0.5)
plt.colorbar(sc, ax=ax, label="log Ie")
ax.set_xlabel("log σ [km/s]")
ax.set_ylabel("log Re [kpc]")
ax.set_title(f"(a) Fundamental Plane ({N} galaxies, RMS={rms_fp:.3f})")

ax = axes[1]
resid = log_Re - pred_fp
ax.scatter(log_sigma, log_Re - pred_virial, s=3, alpha=0.3, c="r", label="Virial")
ax.scatter(log_sigma, resid, s=3, alpha=0.3, c="b", label="Best FP")
ax.axhline(0, color="k", ls="--", lw=0.5)
ax.set_xlabel("log σ [km/s]")
ax.set_ylabel("Residual (dex)")
ax.legend(fontsize=8)
ax.set_title("(b) Residuals: Best FP vs Virial")

ax = axes[2]
ax.scatter(df["n_sersic"], resid, s=3, alpha=0.3, c=log_sigma, cmap="viridis")
r, p = spearmanr(df["n_sersic"], resid)
ax.axhline(0, color="k", ls="--", lw=0.5)
ax.set_xlabel("Sersic n")
ax.set_ylabel("FP residual (dex)")
ax.set_title(f"(c) Residual vs Sersic n (ρ={r:.3f})")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fundamental_plane.pdf", dpi=200)
plt.savefig(f"{OUTDIR}/fundamental_plane.png", dpi=150)
print(f"\n  Saved {OUTDIR}/fundamental_plane.png")
plt.close()

print(f"\n  BEST MODEL: {'Quadratic' if aic_qf < aic_fp else 'Standard'} FP")
print(f"  Virial rejected at ΔAIC = {aic_vir - aic_fp:.0f}")

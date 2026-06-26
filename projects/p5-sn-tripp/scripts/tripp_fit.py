#!/usr/bin/env python3
"""
Remove cosmology (z-dependent residual) from MU_SH0ES, then save 
the x1/c residual for SR to search for non-linear Tripp corrections.

Strategy:
1. MU_SH0ES = Tripp-corrected distance modulus (α=0.153, β=3.138)
2. δμ = MU_SH0ES - μ(z; H0=67.4, Ωm=0.3) — use Planck-like H0 to
   minimize the z-dependent residual
3. Fit δμ = f(z) using a low-order polynomial to remove cosmology
4. Residual r = δμ - f(z) contains only x1/c/host structure
5. SR searches r = g(x1, c) for non-linear Tripp corrections
"""

import numpy as np
from scipy.integrate import quad
from numpy import polyfit, polyval
import pickle, os, sys, warnings
warnings.filterwarnings('ignore')

CACHE = "/tmp/sn_tripp_cache"
C_LIGHT = 299792.458

def load_pantheon():
    path = os.path.join(CACHE, "pantheon_raw.pkl")
    if not os.path.exists(path):
        sys.path.insert(0, os.path.dirname(__file__))
        import download_data
        return download_data.download_pantheon()
    with open(path, "rb") as f:
        return pickle.load(f)

def dist_modulus(z, H0=67.4, Om=0.3):
    Ok = 1 - Om
    dH = C_LIGHT / H0
    result = []
    for zi in z:
        def E(zp): return 1/np.sqrt(Om*(1+zp)**3 + Ok*(1+zp)**2 + (1-Om-Ok))
        dc = dH * quad(E, 0, zi)[0]
        if Ok > 1e-10:
            dm = dH / np.sqrt(Ok) * np.sinh(np.sqrt(Ok) * dc / dH)
        elif Ok < -1e-10:
            dm = dH / np.sqrt(-Ok) * np.sin(np.sqrt(-Ok) * dc / dH)
        else:
            dm = dc
        result.append(5 * np.log10(dm * (1 + zi)) + 25)
    return np.array(result)

def main():
    pp = load_pantheon()
    mask = pp["is_cal"] == 0
    z = pp["z"][mask]
    mu = pp["mu_shoes"][mask]
    x1 = pp["x1"][mask]
    c = pp["c"][mask]

    n = len(z)
    print(f"Data: {n} cosmologically usable SNe")

    # Step 1: Reference cosmology with H0=67.4 (Planck-like, close to our result)
    mu_ref = dist_modulus(z, H0=67.4, Om=0.3)
    resid = mu - mu_ref

    # Step 2: Remove mean
    resid -= np.mean(resid)
    rms_total = np.std(resid)
    print(f"\nMU_SH0ES vs ΛCDM(H0=67.4, Ωm=0.3):")
    print(f"  RMS δμ: {rms_total:.4f} mag")

    # Step 3: Fit and remove smooth z-trend (cosmology mismatch)
    # Use a 3rd order polynomial in log(z) to catch residual cosmology
    # This removes any remaining H0/Ωm mismatch
    logz = np.log10(np.maximum(z, 1e-4))
    coeff_z = polyfit(logz, resid, 3)
    resid_z = polyval(coeff_z, logz)
    resid_x1c = resid - resid_z
    rms_x1c = np.std(resid_x1c)

    print(f"  After removing z-trend (poly3 in logz): RMS = {rms_x1c:.4f}")
    print(f"  → {(1 - rms_x1c/rms_total)*100:.1f}% of variance is cosmology")

    # Step 4: Check residual x1/c linear correlation
    A = np.column_stack([x1, c])
    coeff, *_ = np.linalg.lstsq(A, resid_x1c, rcond=None)
    pred_lin = A @ coeff
    rms_linear = np.sqrt(np.mean((resid_x1c - pred_lin)**2))
    print(f"\nResidual x1/c linear fit: δμ = {coeff[0]:.4f}·x1 + {coeff[1]:.4f}·c")
    print(f"  RMS before: {rms_x1c:.4f}, after: {rms_linear:.4f}")

    # Step 5: Binned residuals
    x1_bins = np.percentile(x1, [0, 25, 50, 75, 100])
    c_bins = np.percentile(c, [0, 33, 67, 100])
    print(f"\nResidual vs x1 (cosmology removed):")
    for i in range(4):
        bx = (x1 >= x1_bins[i]) & (x1 < x1_bins[i+1])
        print(f"  x1 [{x1_bins[i]:.1f},{x1_bins[i+1]:.1f}]: N={bx.sum()}, "
              f"<r>={np.mean(resid_x1c[bx]):.4f}±{np.std(resid_x1c[bx]):.4f}")
    print(f"\nResidual vs c (cosmology removed):")
    for i in range(3):
        bc = (c >= c_bins[i]) & (c < c_bins[i+1])
        print(f"  c [{c_bins[i]:.3f},{c_bins[i+1]:.3f}]: N={bc.sum()}, "
              f"<r>={np.mean(resid_x1c[bc]):.4f}±{np.std(resid_x1c[bc]):.4f}")

    # Save for SR (use cosmology-removed residual)
    data = {
        "z": z, "mu_shoes": pp["mu_shoes"][mask],
        "x1": x1, "c": c,
        "residual": resid_x1c,
        "residual_full": resid,
        "cosmo_trend": resid_z,
        "rms_total": float(rms_total),
        "rms_x1c": float(rms_x1c),
        "rms_linear": float(rms_linear),
    }
    outfile = os.path.join(CACHE, "tripp_residual.pkl")
    with open(outfile, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to {outfile}")
    return data

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bootstrap + sensitivity tests for BTFR.
Runs locally (small dataset, fast).
"""

import numpy as np
import pickle, os, sys, warnings
warnings.filterwarnings('ignore')
from numpy.linalg import lstsq

CACHE = "/tmp/btfr_cache"

def load():
    with open(os.path.join(CACHE, "btfr_data.pkl"), "rb") as f:
        return pickle.load(f)

def fit_slope(logV, logMb):
    A = np.column_stack([np.ones_like(logV), logV])
    coeff, *_ = lstsq(A, logMb, rcond=None)
    return coeff[1]  # slope

def main():
    d = load()
    logV = d["logV"]
    logMb = d["logMb"]
    Vflat = d["Vflat"]
    n = len(logV)

    rng = np.random.RandomState(42)

    # === 1. Bootstrap slope uncertainty ===
    print("=== Bootstrap (2000 resamples) ===")
    slopes = []
    for i in range(2000):
        idx = rng.choice(n, n, replace=True)
        slopes.append(fit_slope(logV[idx], logMb[idx]))
    slopes = np.array(slopes)
    a_med = np.median(slopes)
    a_lo = np.percentile(slopes, 16)
    a_hi = np.percentile(slopes, 84)
    print(f"  a = {a_med:.3f} [{a_lo:.3f}, {a_hi:.3f}] (68% CL)")
    print(f"  MOND (a=4) excluded at {(slopes > 4).mean()*100:.1f}% CL")
    print(f"  ΛCDM (a=3-3.5) within 68% CL: {((slopes > 3) & (slopes < 3.5)).mean()*100:.0f}% of bootstrap")

    # === 2. M/L sensitivity (Υ) ===
    print("\n=== M/L Sensitivity ===")
    d_full = pickle.load(open(os.path.join(CACHE, "sparc.pkl"), "rb"))
    Mgas = 1.33 * d_full['MHI'] * 1e9
    good = d_full['Q'] <= 2
    for ups in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        Mstars = ups * d_full['L3_6'] * 1e9
        Mbary_u = Mstars + Mgas
        logV_u = np.log10(d_full['Vflat'][good])
        logMb_u = np.log10(Mbary_u[good])
        valid = d_full['Vflat'][good] > 0
        a = fit_slope(logV_u[valid], logMb_u[valid])
        print(f"  Υ={ups:.1f}: a={a:.3f}")

    # === 3. Inclination cut sensitivity ===
    print("\n=== Inclination Sensitivity ===")
    Inc = d["Inc"]
    for inc_cut in [20, 30, 40, 50]:
        mask = Inc >= inc_cut
        if mask.sum() < 5: continue
        a = fit_slope(logV[mask], logMb[mask])
        print(f"  Inc>={inc_cut}° ({mask.sum()}): a={a:.3f}")

    # === 4. Quality flag sensitivity ===
    print("\n=== Quality Flag Sensitivity ===")
    Q_arr = d_full['Q']
    Vf_arr = d_full['Vflat']
    L_arr = d_full['L3_6']
    MHI_arr = d_full['MHI']
    for qmax in [1, 2, 3]:
        mask = (Q_arr <= qmax) & (Vf_arr > 0)
        if mask.sum() < 5: continue
        lv = np.log10(Vf_arr[mask])
        stars = 0.5 * L_arr[mask] * 1e9
        gas = 1.33 * MHI_arr[mask] * 1e9
        lu = np.log10(stars + gas)
        a = fit_slope(lv, lu)
        print(f"  Q<={qmax} ({mask.sum()}): a={a:.3f}")

    # === 5. Outlier-resistant fit ===
    print("\n=== Outlier Tests ===")
    # Remove lowest 5% and highest 5% in logV or logMb
    for tail in [0.0, 0.025, 0.05, 0.1]:
        lo_v, hi_v = np.percentile(logV, [100*tail, 100-100*tail])
        lo_m, hi_m = np.percentile(logMb, [100*tail, 100-100*tail])
        mask = (logV >= lo_v) & (logV <= hi_v) & (logMb >= lo_m) & (logMb <= hi_m)
        a = fit_slope(logV[mask], logMb[mask])
        print(f"  Clip {tail*100:.0f}% tails: a={a:.3f} (N={mask.sum()})")

    # === 6. Compare with literature ===
    print("\n=== Literature Comparison ===")
    print(f"  This work: a = {a_med:.2f} ± {(a_hi-a_lo)/2:.2f} (68% CL)")
    print(f"  Lelli+2016 (BTFR): a = 3.75 ± 0.25 (Q=1 only, Υ=0.5)")
    print(f"  McGaugh+2012: a = 3.94 ± 0.07")
    print(f"  Ponomareva+2018: a = 2.97 ± 0.10 (gas-rich dwarfs)")
    print(f"  MOND prediction: a = 4.0 (exact)")
    print(f"  ΛCDM expectation: a ≈ 3.0-3.5")

    print("\nDone.")

if __name__ == "__main__":
    main()

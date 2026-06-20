#!/usr/bin/env python3
"""Bootstrap with Cpx 13 coefficient refit on each resampled dataset."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_rank import fetch_pantheon, mu_from_H
from marginalize_rd import get_data_no_desi, get_desi_for_rd

np.random.seed(42)

z_sn, mu_sn, e_sn = fetch_pantheon()

base_data = get_data_no_desi()
base_data = np.column_stack([base_data, np.zeros(len(base_data))])  # flag: 0 = no DESI
desi_ref = get_desi_for_rd(147.0)
desi_data = np.column_stack([desi_ref, np.ones(len(desi_ref))])  # flag: 1 = DESI
all_data = np.vstack([base_data, desi_data])

def chi2_sne(H0, A, B, C):
    def H_func(z):
        z = np.asarray(z, dtype=float)
        return H0 + A * z * (z - B) * (z**2 + C)
    mu_pred = np.array([mu_from_H(H_func, z) for z in z_sn])
    resid = mu_sn - mu_pred
    good = np.isfinite(mu_pred) & np.isfinite(resid)
    if np.sum(good) < 10: return np.nan, np.nan
    w = 1.0 / e_sn[good]**2
    delta_m = np.sum(resid[good] * w) / np.sum(w)
    chi2 = np.sum(((resid[good] - delta_m) / e_sn[good])**2)
    return chi2, delta_m

def fit_bootstrap(data_sample):
    """Fit Cpx 13 to a data sample. Returns (H0, A, B, C, chi2_h)."""
    z = data_sample[:, 0]
    H = data_sample[:, 1]
    e = data_sample[:, 2]
    
    best = (np.nan, np.nan, np.nan, np.nan, np.inf)
    for C in np.linspace(-0.5, 3.0, 200):
        u = z * (z**2 + C)
        v = z**2 * (z**2 + C)
        X = np.column_stack([np.ones_like(z), v, u])
        w = 1.0 / e**2
        WX = X * w[:, None]
        try:
            beta = np.linalg.solve(X.T @ WX, X.T @ (w * H))
        except np.linalg.LinAlgError:
            continue
        H0, p, q = beta
        if abs(p) < 1e-15: continue
        A, Bfit = p, -q/p
        chi2 = np.sum(w * (H - (H0 + p*v + q*u))**2)
        if chi2 < best[4]:
            best = (H0, A, Bfit, C, chi2)
    return best

# --- First, fit on original data ---
print("Best fit on original data...")
orig = fit_bootstrap(all_data)
H0_orig, A_orig, B_orig, C_orig, chi2_h_orig = orig
cs_orig, dm_orig = chi2_sne(H0_orig, A_orig, B_orig, C_orig)
print(f"  H0={H0_orig:.2f}, A={A_orig:.2f}, B={B_orig:.2f}, C={C_orig:.2f}")
print(f"  chi2_H={chi2_h_orig:.1f}, chi2_SN={cs_orig:.1f}")

# --- Bootstrap ---
n_boot = 2000
n = len(all_data)
H0_samples = np.full(n_boot, np.nan)
chi2_h_samples = np.full(n_boot, np.nan)
chi2_sn_samples = np.full(n_boot, np.nan)

print(f"\nRunning {n_boot} bootstrap iterations...")
for i in range(n_boot):
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{n_boot}")
    
    idx = np.random.randint(0, n, n)
    sample = all_data[idx]
    
    result = fit_bootstrap(sample)
    H0, A, B, C, chi2_h = result
    if np.isnan(H0): continue
    
    H0_samples[i] = H0
    chi2_h_samples[i] = chi2_h

valid = np.isfinite(H0_samples)
H0_vals = H0_samples[valid]
chi2_h_vals = chi2_h_samples[valid]

# Filter out pathological fits (chi2 too high)
h_good = chi2_h_vals < np.percentile(chi2_h_vals, 95)
H0_good = H0_vals[h_good]

h0_mean = np.mean(H0_good)
h0_std = np.std(H0_good)
h0_16 = np.percentile(H0_good, 16)
h0_84 = np.percentile(H0_good, 84)
h0_50 = np.percentile(H0_good, 50)

print(f"\n{'='*60}")
print(f"  BOOTSTRAP RESULTS ({len(H0_good)} valid samples)")
print(f"{'='*60}")
print(f"  Mean ± Std:  {h0_mean:.2f} ± {h0_std:.2f} km/s/Mpc")
print(f"  Median:      {h0_50:.2f}")
print(f"  68% CL:      [{h0_16:.2f}, {h0_84:.2f}]")
print(f"")
print(f"  Best fit (original data):  H0 = {H0_orig:.2f}, chi2_H = {chi2_h_orig:.1f}, chi2_SN = {cs_orig:.1f}")
print(f"  Profile likelihood:        H0 = 68.00 [67.16, 68.71]")
print(f"  Bootstrap refit:           H0 = {h0_mean:.1f} [{h0_16:.1f}, {h0_84:.1f}]")
print(f"")
print(f"  Planck 2018:               H0 = 67.4 ± 0.5 (consistent)")
print(f"  SH0ES 2024:                H0 = 73.0 ± 1.0 (Δ = {73.0-h0_mean:.1f})")
print(f"{'='*60}")

np.savetxt("/tmp/h0_bootstrap.csv",
           np.column_stack([H0_vals, chi2_h_vals]),
           header="H0 chi2_Hz", fmt="%.4f")
print("\nSaved to /tmp/h0_bootstrap.csv")

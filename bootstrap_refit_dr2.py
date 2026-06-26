#!/usr/bin/env python3
"""Bootstrap with Cpx 13 coefficient refit, DESI DR2."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_hz, fetch_pantheon, mu_from_H

np.random.seed(42)
hz_data = load_hz(version='dr2')
zh, Hh, eh = hz_data[:,0], hz_data[:,1], hz_data[:,2]
n = len(hz_data)

z_sn, mu_sn, e_sn = fetch_pantheon()

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

def fit_bootstrap(z, H, e):
    best = (np.nan, np.nan, np.nan, np.nan, np.inf)
    for C in np.linspace(-0.5, 3.0, 200):
        u = z * (z**2 + C)
        v = z**2 * (z**2 + C)
        X = np.column_stack([np.ones_like(z), v, u])
        w = 1.0 / e**2
        try:
            beta = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * H))
        except np.linalg.LinAlgError:
            continue
        H0, p, q = beta
        if abs(p) < 1e-15: continue
        A, Bfit = p, -q/p
        chi2 = np.sum(w * (H - (H0 + p*v + q*u))**2)
        if chi2 < best[4]:
            best = (H0, A, Bfit, C, chi2)
    return best

print("Best fit on original data (DR2)...")
orig = fit_bootstrap(zh, Hh, eh)
H0_orig, A_orig, B_orig, C_orig, chi2_h_orig = orig
cs_orig, dm_orig = chi2_sne(H0_orig, A_orig, B_orig, C_orig)
print(f"  H0={H0_orig:.2f}, A={A_orig:.2f}, B={B_orig:.2f}, C={C_orig:.2f}")
print(f"  chi2_H={chi2_h_orig:.1f}, chi2_SN={cs_orig:.1f}")

n_boot = 2000
H0_samples = np.full(n_boot, np.nan)
chi2_h_samples = np.full(n_boot, np.nan)

print(f"\nRunning {n_boot} bootstrap iterations (DR2)...")
for i in range(n_boot):
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{n_boot}")
    idx = np.random.randint(0, n, n)
    result = fit_bootstrap(zh[idx], Hh[idx], eh[idx])
    H0, A, B, C, chi2_h = result
    if np.isnan(H0): continue
    H0_samples[i] = H0
    chi2_h_samples[i] = chi2_h

valid = np.isfinite(H0_samples)
H0_vals = H0_samples[valid]
chi2_h_vals = chi2_h_samples[valid]

h_good = chi2_h_vals < np.percentile(chi2_h_vals, 95)
H0_good = H0_vals[h_good]

h0_mean = np.mean(H0_good)
h0_std = np.std(H0_good)
h0_16 = np.percentile(H0_good, 16)
h0_84 = np.percentile(H0_good, 84)

print(f"\n{'='*60}")
print(f"  BOOTSTRAP RESULTS (DR2, {len(H0_good)} valid samples)")
print(f"{'='*60}")
print(f"  Mean ± Std:  {h0_mean:.2f} ± {h0_std:.2f} km/s/Mpc")
print(f"  68% CL:      [{h0_16:.2f}, {h0_84:.2f}]")
print(f"")
print(f"  Best fit (original data):  H0 = {H0_orig:.2f}, chi2_H = {chi2_h_orig:.1f}")
print(f"  Profile likelihood:        H0 = 68.00 [67.3, 69.1]")
print(f"  Bootstrap refit:           H0 = {h0_mean:.1f} [{h0_16:.1f}, {h0_84:.1f}]")
print(f"{'='*60}")

np.savetxt("/tmp/h0_bootstrap_dr2.csv",
           np.column_stack([H0_vals, chi2_h_vals]),
           header="H0 chi2_Hz (DR2)", fmt="%.4f")
print("\nSaved to /tmp/h0_bootstrap_dr2.csv")

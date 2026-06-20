#!/usr/bin/env python3
"""Joint H0 profile from Cpx 13 + free H0, scanning (H0, C) with analytical (A,B)."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_rank import load_data, fetch_pantheon, mu_from_H

hz = load_data()
z_h, H_h, e_h = hz[:, 0], hz[:, 1], hz[:, 2]
z_sn, mu_sn, e_sn = fetch_pantheon()

def fit_AB(H0, C):
    u = z_h * (z_h**2 + C)
    v = z_h**2 * (z_h**2 + C)
    y = H_h - H0
    w = 1.0 / e_h**2
    X = np.column_stack([v, u])
    WX = X * w[:, None]
    try:
        sol = np.linalg.solve(X.T @ WX, X.T @ (w * y))
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.inf
    p, q = sol
    if abs(p) < 1e-15:
        return np.nan, np.nan, np.inf
    A, B = p, -q/p
    chi2_h = np.sum(w * (y - (p*v + q*u))**2)
    return A, B, chi2_h

def chi2_sne(A, B, C, H0):
    def H_func(z):
        z = np.asarray(z, dtype=float)
        return H0 + A * z * (z - B) * (z**2 + C)
    mu_pred = np.array([mu_from_H(H_func, z) for z in z_sn])
    resid = mu_sn - mu_pred
    good = np.isfinite(mu_pred) & np.isfinite(resid)
    if np.sum(good) < 10: return np.nan
    w = 1.0 / e_sn[good]**2
    return np.sum(((resid[good] - np.sum(resid[good]*w)/np.sum(w)) / e_sn[good])**2)

# --- Grid scan: H0 × C, analytically fit A,B ---
print("Scanning H0 × C grid (joint CC+BAO+SNe)...")
H0_grid = np.arange(60.0, 80.01, 0.5)
C_grid = np.arange(-1.0, 10.01, 0.5)

results_joint = np.full((len(H0_grid), len(C_grid)), np.nan)
results_hz = np.full((len(H0_grid), len(C_grid)), np.nan)
best_params = None
best_joint = np.inf

n_total = len(H0_grid) * len(C_grid)
n_done = 0
for i, H0 in enumerate(H0_grid):
    for j, C in enumerate(C_grid):
        A, B, chi2_h = fit_AB(H0, C)
        if not np.isfinite(chi2_h) or chi2_h > 100:
            continue
        results_hz[i, j] = chi2_h
        chi2_s = chi2_sne(A, B, C, H0)
        if not np.isfinite(chi2_s):
            continue
        chi2_j = chi2_h + chi2_s
        results_joint[i, j] = chi2_j
        if chi2_j < best_joint:
            best_joint = chi2_j
            best_params = (H0, A, B, C, chi2_h, chi2_s)
        n_done += 1
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(H0_grid)} H0 values done ({n_done} grid points evaluated)")

# --- H0 profile: marginalize over (B,C) by taking minimum at each H0 ---
joint_profile = np.nanmin(results_joint, axis=1)
hz_profile = np.nanmin(results_hz, axis=1)

joint_profile -= np.nanmin(joint_profile)
hz_profile -= np.nanmin(hz_profile)

def crossing(h, d, target):
    valid = np.isfinite(d)
    h, d = h[valid], d[valid]
    above = d >= target
    if np.sum(above) < 2 or np.sum(~above) < 2: return None, None
    i = np.where(above[:-1] != above[1:])[0]
    if len(i) == 0: return None, None
    vals = [h[ix] + (target-d[ix])/(d[ix+1]-d[ix])*(h[ix+1]-h[ix]) for ix in i]
    return min(vals), max(vals)

h0_j_ml = H0_grid[np.nanargmin(joint_profile)]
h0_h_ml = H0_grid[np.nanargmin(hz_profile)]
lo_j1, hi_j1 = crossing(H0_grid, joint_profile, 1.0)
lo_j2, hi_j2 = crossing(H0_grid, joint_profile, 4.0)
lo_h1, hi_h1 = crossing(H0_grid, hz_profile, 1.0)

H0_best, A_best, B_best, C_best, ch_best, cs_best = best_params

print(f"\n{'='*60}")
print(f"  RESULTS: Cpx 13 + free H0")
print(f"{'='*60}")
print(f"  Form: H(z) = H0 + A*z*(z-B)*(z\u00b2+C)")
print(f"")
print(f"  Joint best fit:")
print(f"    H0     = {H0_best:.1f} km/s/Mpc")
print(f"    A      = {A_best:.2f}")
print(f"    B      = {B_best:.2f}")
print(f"    C      = {C_best:.2f}")
print(f"    chi2_H = {ch_best:.1f}")
print(f"    chi2_S = {cs_best:.1f}")
print(f"    total  = {best_joint:.1f}")
print(f"")
print(f"  Joint H0 profile:")
if lo_j1: print(f"    68% CL = [{lo_j1:.1f}, {hi_j1:.1f}]")
if lo_j2: print(f"    95% CL = [{lo_j2:.1f}, {hi_j2:.1f}]")
print(f"")
print(f"  CC+BAO only H0 profile:")
print(f"    Best: H0 = {h0_h_ml:.1f}")
if lo_h1: print(f"    68% CL = [{lo_h1:.1f}, {hi_h1:.1f}]")
print(f"")
print(f"  SR-discovered Cpx 13 (f(0)=0): H0 = 67.4")
print(f"  Planck 2018:   H0 = 67.4 ± 0.5")
print(f"  SH0ES 2024:    H0 = 73.0 ± 1.0")
print(f"{'='*60}")

# Save
np.savetxt("/tmp/h0_joint_profile.csv",
           np.column_stack([H0_grid, joint_profile, hz_profile]),
           header="H0 deltaJoint deltaHz", fmt="%.4f")
print("\nSaved to /tmp/h0_joint_profile.csv")

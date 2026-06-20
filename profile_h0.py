#!/usr/bin/env python3
"""Profile likelihood for H0 (no scipy dependency)."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_rank import load_data, fetch_pantheon, mu_from_H

hz = load_data()
z_h, H_h, e_h = hz[:, 0], hz[:, 1], hz[:, 2]
z_sn, mu_sn, e_sn = fetch_pantheon()

def H_model(z, H0, A, B, C):
    return H0 + A * z * (z - B) * (z**2 + C)

def chi2_sn(params):
    H0, A, B, C = params
    def H_func(z):
        return H_model(np.asarray(z, dtype=float), H0, A, B, C)
    mu_pred = np.array([mu_from_H(H_func, z) for z in z_sn])
    resid = mu_sn - mu_pred
    good = np.isfinite(mu_pred) & np.isfinite(resid)
    if np.sum(good) < 10: return np.nan
    w = 1.0 / e_sn[good]**2
    return np.sum(((resid[good] - np.sum(resid[good]*w)/np.sum(w)) / e_sn[good])**2)

def fit_Hz_fixed_H0_C(H0, C):
    """Fit A, B analytically for fixed H0, C using linear least squares.
    
    H(z) - H0 = A*z*(z-B)*(z^2+C)
              = A*z^2*(z^2+C) - A*B*z*(z^2+C)
              = p*v(z) + q*u(z)
    where p = A, q = -A*B, u = z*(z^2+C), v = z^2*(z^2+C)
    """
    u = z_h * (z_h**2 + C)
    v = z_h**2 * (z_h**2 + C)
    y = H_h - H0
    w = 1.0 / e_h**2
    
    X = np.column_stack([v, u])
    WX = X * w[:, None]
    lhs = X.T @ WX
    rhs = X.T @ (w * y)
    
    try:
        sol = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None, None, np.inf
    
    p, q = sol
    A = p
    if abs(p) < 1e-15:  # degenerate
        B = 0
    else:
        B = -q / p
    
    chi2 = np.nansum(w * (y - (p*v + q*u))**2)
    return A, B, chi2

def fit_Hz_fixed_H0(H0, C_grid=np.linspace(-5, 10, 301)):
    """Find best A,B,C for a given H0 by scanning C."""
    best_chi2 = np.inf
    best_ABC = None
    for C in C_grid:
        A, B, chi2 = fit_Hz_fixed_H0_C(H0, C)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_ABC = (A, B, C)
    return best_ABC, best_chi2

# --- Global fit (CC+BAO) ---
print("Fitting CC+BAO...")
# Coarse grid scan first
print("  Global grid scan...")
best_chi2 = np.inf
best_params = None
for H0 in np.arange(55, 82, 1.0):
    (A, B, C), chi2 = fit_Hz_fixed_H0(H0)
    if chi2 < best_chi2:
        best_chi2 = chi2
        best_params = (H0, A, B, C)

# Refine with finer grid around best
print(f"  Refining around H0={best_params[0]:.1f}...")
H0_centered = np.arange(best_params[0]-1.5, best_params[0]+1.51, 0.1)
for H0 in H0_centered:
    (A, B, C), chi2 = fit_Hz_fixed_H0(H0)
    if chi2 < best_chi2:
        best_chi2 = chi2
        best_params = (H0, A, B, C)

h0_ml, A_ml, B_ml, C_ml = best_params
ch_min = best_chi2
print(f"  H0={h0_ml:.2f}, A={A_ml:.4f}, B={B_ml:.4f}, C={C_ml:.4f}, chi2={ch_min:.1f}")

# --- H0 profile ---
print("\nScanning H0 profile...")
H0_grid = np.arange(55, 82.01, 0.2)
d_hz = np.full(len(H0_grid), np.nan)
ABC_best = [None] * len(H0_grid)
for i, h0 in enumerate(H0_grid):
    abc, chi2 = fit_Hz_fixed_H0(h0)
    d_hz[i] = chi2
    ABC_best[i] = abc

d_hz_min = np.nanmin(d_hz)
d_hz -= d_hz_min

# Crossing
def crossing(h, d, target):
    valid = np.isfinite(d)
    h, d = h[valid], d[valid]
    above = d >= target
    if np.sum(above) < 2 or np.sum(~above) < 2: return None, None
    i = np.where(above[:-1] != above[1:])[0]
    if len(i) == 0: return None, None
    vals = [h[ix] + (target-d[ix])/(d[ix+1]-d[ix])*(h[ix+1]-h[ix]) for ix in i]
    return min(vals), max(vals)

lo1, hi1 = crossing(H0_grid, d_hz, 1.0)
lo2, hi2 = crossing(H0_grid, d_hz, 4.0)

# SNe chi2 at best fit
print("Computing SNe chi2 at best fit...")
cs = chi2_sn(best_params)
print(f"  Done. chi2_SN = {cs:.1f}")

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  Cpx 13 form: H(z) = H0 + A*z*(z-B)*(z\u00b2+C)")
print(f"  Fit to CC+BAO (36 pts, 4 params, dof=32)")
print(f"")
print(f"  H0   = {h0_ml:.1f} km/s/Mpc")
if lo1: print(f"  68%  = [{lo1:.1f}, {hi1:.1f}]")
if lo2: print(f"  95%  = [{lo2:.1f}, {hi2:.1f}]")
print(f"  A    = {A_ml:.2f}")
print(f"  B    = {B_ml:.2f}")
print(f"  C    = {C_ml:.2f}")
print(f"")
print(f"  chi2_Hz  = {ch_min:.1f} (dof=32, reduced={ch_min/32:.2f})")
print(f"  chi2_SN  = {cs:.1f} (1590 pts, free M)")
print(f"")
print(f"  Planck 2018:  H0 = 67.4 ± 0.5")
print(f"  SH0ES 2024:   H0 = 73.0 ± 1.0")
print(f"{'='*60}")

np.savetxt("/tmp/h0_profile.csv",
           np.column_stack([H0_grid, d_hz]),
           header="H0 dHz", fmt="%.4f")
print("Saved to /tmp/h0_profile.csv")

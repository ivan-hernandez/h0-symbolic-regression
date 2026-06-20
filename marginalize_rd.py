#!/usr/bin/env python3
"""H0 profile with r_d marginalization for DESI BAO."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_rank import load_data, fetch_pantheon, mu_from_H

# --- Load data (CC + SDSS BAO only — DESI added separately with r_d) ---
def get_data_no_desi():
    cc = np.array([
        [0.070,69.0,19.6],[0.090,69.0,12.0],[0.120,68.6,26.2],[0.170,83.0,8.0],
        [0.1791,75.0,4.0],[0.1993,75.0,5.0],[0.200,72.9,29.6],[0.270,77.0,14.0],
        [0.280,88.8,36.6],[0.3519,83.0,14.0],[0.3802,83.0,13.5],[0.400,95.0,17.0],
        [0.4004,77.0,10.2],[0.4247,87.1,11.2],[0.4497,92.8,12.9],[0.470,89.0,34.0],
        [0.4783,80.9,9.0],[0.480,97.0,62.0],[0.5929,104.0,13.0],[0.6797,92.0,8.0],
        [0.750,98.8,33.6],[0.7812,105.0,12.0],[0.800,113.1,28.5],[0.8754,125.0,17.0],
        [0.880,90.0,40.0],[0.900,117.0,23.0],[1.037,154.0,20.0],[1.300,168.0,17.0],
        [1.363,160.0,33.6],[1.430,177.0,18.0],[1.530,140.0,14.0],[1.750,202.0,40.0],
        [1.965,186.5,50.4],
    ])
    bao_sdss = np.array([[0.380,81.1,2.2],[0.510,91.1,2.1],[0.610,99.4,2.2]])
    combined = np.vstack([cc, bao_sdss])
    combined = combined[combined[:, 0].argsort()]
    return combined

def get_desi_for_rd(rd):
    c = 299792.458
    desi = np.array([
        [0.510, 20.98334647, 0.61],
        [0.706, 20.07872919, 0.60],
        [0.930, 17.87612922, 0.35],
        [1.317, 13.82372285, 0.42],
        [2.330,  8.52256583, 0.17],
    ])
    hz = c / (rd * desi[:, 1])
    errs = hz * desi[:, 2] / desi[:, 1]
    return np.column_stack([desi[:, 0], hz, errs])

z_sn, mu_sn, e_sn = fetch_pantheon()

base_data = get_data_no_desi()
z_base, H_base, e_base = base_data[:, 0], base_data[:, 1], base_data[:, 2]

def fit_parameters_at_H0_rd(H0, rd):
    """Fit A, B, C at fixed H0, rd using analytical scan over C."""
    desi = get_desi_for_rd(rd)
    z_all = np.concatenate([z_base, desi[:, 0]])
    H_all = np.concatenate([H_base, desi[:, 1]])
    e_all = np.concatenate([e_base, desi[:, 2]])
    
    def fit_AB(H0, C, z, H, e):
        u = z * (z**2 + C)
        v = z**2 * (z**2 + C)
        y = H - H0
        w = 1.0 / e**2
        X = np.column_stack([v, u])
        try:
            sol = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.inf
        p, q = sol
        if abs(p) < 1e-15: return np.nan, np.nan, np.inf
        return p, -q/p, np.sum(w * (y - (p*v + q*u))**2)
    
    best_chi2 = np.inf
    best_ABC = None
    for C in np.linspace(-0.5, 3.0, 150):
        A, B, chi2 = fit_AB(H0, C, z_all, H_all, e_all)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_ABC = (A, B, C)
    return best_ABC, best_chi2

def chi2_sne_at_params(H0, A, B, C):
    def H_func(z):
        z = np.asarray(z, dtype=float)
        return H0 + A * z * (z - B) * (z**2 + C)
    mu_pred = np.array([mu_from_H(H_func, z) for z in z_sn])
    resid = mu_sn - mu_pred
    good = np.isfinite(mu_pred) & np.isfinite(resid)
    if np.sum(good) < 10: return np.nan
    w = 1.0 / e_sn[good]**2
    return np.sum(((resid[good] - np.sum(resid[good]*w)/np.sum(w)) / e_sn[good])**2)

# --- 2D scan: H0 × r_d ---
print("Scanning H0 × r_d...")
H0_grid = np.concatenate([np.arange(55, 65.25, 0.5), 
                           np.arange(65, 70.25, 0.2),
                           np.arange(70, 82.25, 0.5)])
rd_grid = np.array([146.0, 146.5, 147.0, 147.5, 148.0])
rd_prior_mu, rd_prior_sigma = 147.09, 0.26

chi2_joint = np.full((len(H0_grid), len(rd_grid)), np.nan)
best_params = {}

for j, rd in enumerate(rd_grid):
    for i, H0 in enumerate(H0_grid):
        abc, chi2_h = fit_parameters_at_H0_rd(H0, rd)
        if abc is None or np.isinf(chi2_h): continue
        A, B, C = abc
        chi2_s = chi2_sne_at_params(H0, A, B, C)
        if np.isnan(chi2_s): continue
        chi2_joint[i, j] = chi2_h + chi2_s
    print(f"  r_d = {rd:.1f} done")

# Marginalize over r_d with Gaussian prior
# Weight each H0 value by the minimum over r_d + prior penalty
chi2_marginal = np.full(len(H0_grid), np.nan)
for i, H0 in enumerate(H0_grid):
    best = np.inf
    for j, rd in enumerate(rd_grid):
        if np.isnan(chi2_joint[i, j]): continue
        rd_prior = ((rd - rd_prior_mu) / rd_prior_sigma)**2
        total = chi2_joint[i, j] + rd_prior
        if total < best:
            best = total
    chi2_marginal[i] = best

chi2_marginal -= np.nanmin(chi2_marginal)

def crossing(h, d, target):
    valid = np.isfinite(d)
    h, d = h[valid], d[valid]
    above = d >= target
    if np.sum(above) < 2: return None, None
    i = np.where(above[:-1] != above[1:])[0]
    if len(i) == 0: return None, None
    vals = [h[ix] + (target-d[ix])/(d[ix+1]-d[ix])*(h[ix+1]-h[ix]) for ix in i]
    return min(vals), max(vals)

lo1, hi1 = crossing(H0_grid, chi2_marginal, 1.0)
lo2, hi2 = crossing(H0_grid, chi2_marginal, 4.0)
h0_ml = H0_grid[np.nanargmin(chi2_marginal)]

print(f"\n{'='*60}")
print(f"  H0 with r_d marginalized")
print(f"{'='*60}")
print(f"  H0     = {h0_ml:.2f} km/s/Mpc")
if lo1: print(f"  68% CL = [{lo1:.2f}, {hi1:.2f}]")
if lo2: print(f"  95% CL = [{lo2:.2f}, {hi2:.2f}]")
print(f"")
print(f"  r_d prior: {rd_prior_mu} ± {rd_prior_sigma} Mpc (Planck 2018)")
print(f"  Marginalized over r_d ∈ [{rd_grid[0]:.1f}, {rd_grid[-1]:.1f}]")
print(f"")
print(f"  Without r_d marginalization: 67.80 [66.88, 68.74]")
print(f"  With r_d marginalization:    {h0_ml:.2f} [{lo1:.2f}, {hi1:.2f}]")
print(f"{'='*60}")

# Also report the fixed r_d = 147 result from this code
print(f"\n  Fixed r_d = 147 (for comparison):")
for i, H0 in enumerate(H0_grid):
    if not np.isnan(chi2_joint[i, rd_grid.tolist().index(147.0)]):
        pass
chi2_fixed = chi2_joint[:, rd_grid.tolist().index(147.0)]
chi2_fixed -= np.nanmin(chi2_fixed)
l1, h1 = crossing(H0_grid, chi2_fixed, 1.0)
h0_fixed = H0_grid[np.nanargmin(chi2_fixed)]
print(f"    H0 = {h0_fixed:.2f} [{l1:.2f}, {h1:.2f}]")

np.savetxt("/tmp/h0_with_rd.csv",
           np.column_stack([H0_grid, chi2_marginal]),
           header="H0 deltaChi2 (r_d marginalized)", fmt="%.4f")
print("\nSaved to /tmp/h0_with_rd.csv")

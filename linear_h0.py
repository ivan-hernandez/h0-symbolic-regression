#!/usr/bin/env python3
"""H0 from linear fit of quartic H(z) + SNe consistency check."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_rank import load_data, fetch_pantheon, mu_from_H

hz = load_data()
z_h, H_h, e_h = hz[:, 0], hz[:, 1], hz[:, 2]
z_sn, mu_sn, e_sn = fetch_pantheon()

# Weighted linear least squares: H(z) = H0 + p1*z + p2*z^2 + p3*z^3 + p4*z^4
def fit_quartic():
    w = 1.0 / e_h**2
    X = np.column_stack([np.ones_like(z_h), z_h, z_h**2, z_h**3, z_h**4])
    WX = X * w[:, None]
    lhs = X.T @ WX
    rhs = X.T @ (w * H_h)
    beta = np.linalg.solve(lhs, rhs)
    cov = np.linalg.inv(lhs)
    err = np.sqrt(np.diag(cov))
    chi2 = np.sum(w * (H_h - X @ beta)**2)
    return beta, err, chi2, X

print("Fitting H(z) = H0 + p1*z + p2*z^2 + p3*z^3 + p4*z^4")
beta, err, chi2, X = fit_quartic()
dof = len(z_h) - len(beta)

names = ['H0', 'p1', 'p2', 'p3', 'p4']
print(f"\n  chi2 = {chi2:.1f}, dof = {dof}, reduced chi2 = {chi2/dof:.2f}")
print(f"  Q (p-value) = {1 - np.sum(np.random.chisquare(dof, 100000) < chi2)/100000:.3f}")  # approximate

print(f"\n  Parameter   Value     Error")
for n, b, e in zip(names, beta, err):
    print(f"  {n:>10} = {b:6.2f} ± {e:5.2f}")

# Covariance matrix (display as correlation)
corr = cov / np.outer(err, err)
print(f"\n  Correlation matrix:")
for i, n in enumerate(names):
    print(f"    {n:>4}: " + " ".join(f"{corr[i,j]:+.2f}" for j in range(len(names))))

# H0 uncertainty (which is what we care about)
print(f"\n{'='*60}")
print(f"  H0 = {beta[0]:.2f} ± {err[0]:.2f} km/s/Mpc  (68% CL from quartic fit)")
print(f"{'='*60}")

# Evaluate SNe chi2 at best-fit quartic
def make_H_func(params):
    def H_func(z):
        z = np.asarray(z, dtype=float)
        return params[0] + params[1]*z + params[2]*z**2 + params[3]*z**3 + params[4]*z**4
    return H_func

print("\nComputing SNe chi2 at quartic best fit...")
H_func = make_H_func(beta)
mu_pred = np.array([mu_from_H(H_func, z) for z in z_sn])
resid = mu_sn - mu_pred
good = np.isfinite(mu_pred) & np.isfinite(resid)
w = 1.0 / e_sn[good]**2
delta_m = np.sum(resid[good] * w) / np.sum(w)
chi2_mu = np.sum(((resid[good] - delta_m) / e_sn[good])**2)
print(f"  chi2_SN = {chi2_mu:.1f} (1590 pts, free M, ΔM = {delta_m:.3f})")

# Compare with LCDM
print(f"\n  ΛCDM reference:")
print(f"  chi2_H(LCDM)  = 16.6 (from CC+BAO)")
print(f"  chi2_SN(LCDM) = 688.0 (1590 pts, free M)")

# Plot profile (quick ASCII visualization)
print(f"\n  H0 profile (Δchi2):")
h0_vals = np.arange(55, 82.5, 0.5)
delta = np.zeros_like(h0_vals)
for i, h0fix in enumerate(h0_vals):
    # Refit with H0 fixed
    w = 1.0 / e_h**2
    Xr = np.column_stack([z_h, z_h**2, z_h**3, z_h**4])
    y = H_h - h0fix
    WX = Xr * w[:, None]
    lhs = Xr.T @ WX
    rhs = Xr.T @ (w * y)
    beta_r = np.linalg.solve(lhs, rhs)
    chi2_r = np.sum(w * (y - Xr @ beta_r)**2)
    delta[i] = chi2_r - chi2

min_idx = np.argmin(delta)
h0_ml = h0_vals[min_idx]
delta_min = delta[min_idx]
delta -= delta_min

# Crossing
def crossing(h, d, target):
    valid = np.isfinite(d)
    h, d = h[valid], d[valid]
    above = d >= target
    i = np.where(above[:-1] != above[1:])[0]
    if len(i) == 0: return None, None
    vals = [h[ix] + (target-d[ix])/(d[ix+1]-d[ix])*(h[ix+1]-h[ix]) for ix in i]
    return min(vals), max(vals)

lo1, hi1 = crossing(h0_vals, delta, 1.0)
lo2, hi2 = crossing(h0_vals, delta, 4.0)

print(f"  H0 (MLE)  = {h0_ml:.1f}")
if lo1: print(f"  68% CL    = [{lo1:.1f}, {hi1:.1f}]")
if lo2: print(f"  95% CL    = [{lo2:.1f}, {hi2:.1f}]")

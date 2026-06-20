#!/usr/bin/env python3
"""Bootstrap H0 uncertainty from joint CC+BAO+SNe analysis."""

import numpy as np
import csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_rank import load_data, fetch_pantheon, make_H_func, mu_from_H

np.random.seed(42)

hof_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/seed_hofs/20260620_111649_mX2UPk.csv"

print("Loading data...")
data_hz = load_data()
z_h, H_h, e_h = data_hz[:, 0], data_hz[:, 1], data_hz[:, 2]
z_sn, mu_sn, e_sn = fetch_pantheon()
print(f"  CC+BAO: {len(z_h)} pts, SNe: {len(z_sn)} pts")

print("Loading HoF...")
equations = []
with open(hof_file) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        equations.append((int(row[0]), float(row[1]), row[2]))
print(f"  {len(equations)} expressions")

print("Precomputing SNe chi2 and H_funcs...")
sn_chi2 = np.full(len(equations), np.nan)
H_funcs = [None] * len(equations)
f_zero = np.full(len(equations), np.nan)
for idx, (cpx, loss, eq) in enumerate(equations):
    try:
        H_func = make_H_func(eq)
        H_funcs[idx] = H_func
        f_zero[idx] = H_func(np.array([0.0]))[0] - 67.4
        mu_pred = np.array([mu_from_H(H_func, z) for z in z_sn])
        resid = mu_sn - mu_pred
        good = np.isfinite(mu_pred) & np.isfinite(resid)
        if np.sum(good) > 10:
            w = 1.0 / e_sn[good]**2
            delta_m = np.sum(resid[good] * w) / np.sum(w)
            sn_chi2[idx] = np.sum(((resid[good] - delta_m) / e_sn[good])**2)
    except Exception as e:
        pass

print(f"  {np.sum(np.isfinite(sn_chi2))} valid SNe chi2")

n_boot = 2000
n = len(data_hz)
H0_vals = np.empty(n_boot)
best_idx = np.empty(n_boot, dtype=int)

print(f"Running {n_boot} bootstrap iterations...")
for i in range(n_boot):
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{n_boot}")
    
    idx = np.random.randint(0, n, n)
    z_b, H_b, e_b = z_h[idx], H_h[idx], e_h[idx]
    
    best_joint = np.inf
    best = -1
    best_h0 = np.nan
    
    for j, (cpx, loss, eq) in enumerate(equations):
        H_func = H_funcs[j]
        if H_func is None or np.isnan(sn_chi2[j]):
            continue
        
        try:
            H_pred = H_func(z_b)
            chi2_h = np.nansum(((H_b - H_pred) / e_b)**2)
            joint = chi2_h + sn_chi2[j]
        except:
            continue
        
        if joint < best_joint:
            best_joint = joint
            best = j
            best_h0 = 67.4 + f_zero[j]
    
    H0_vals[i] = best_h0
    best_idx[i] = best

h0_mean = np.mean(H0_vals)
h0_std = np.std(H0_vals)
h0_16 = np.percentile(H0_vals, 16)
h0_84 = np.percentile(H0_vals, 84)
h0_50 = np.percentile(H0_vals, 50)

print(f"\n{'='*60}")
print(f"  H0 bootstrap results ({n_boot} iterations):")
print(f"  Mean ± Std:  {h0_mean:.2f} ± {h0_std:.2f} km/s/Mpc")
print(f"  Median:      {h0_50:.2f} km/s/Mpc")
print(f"  16%ile:      {h0_16:.2f} km/s/Mpc")
print(f"  84%ile:      {h0_84:.2f} km/s/Mpc")
print(f"  68% CL:      [{h0_16:.2f}, {h0_84:.2f}]")
print(f"{'='*60}")

# How often each expression wins
unique, counts = np.unique(best_idx, return_counts=True)
print(f"\n  Model selection frequency:")
for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
    cpx, _, eq = equations[u]
    pct = 100 * c / n_boot
    h0 = 67.4 + f_zero[u]
    print(f"    Cpx {cpx:>2}: {pct:>5.1f}%  H0={h0:.1f}  eq={eq[:50]}")

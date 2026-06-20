#!/usr/bin/env python3
"""H0 result: profile + plot."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_rank import load_data, fetch_pantheon, mu_from_H

hz = load_data()
z_h, H_h, e_h = hz[:, 0], hz[:, 1], hz[:, 2]
z_sn, mu_sn, e_sn = fetch_pantheon()

def fit_AB(H0, C):
    u = z_h * (z_h**2 + C); v = z_h**2 * (z_h**2 + C)
    y = H_h - H0; w = 1.0 / e_h**2
    X = np.column_stack([v, u])
    sol = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
    p, q = sol
    if abs(p) < 1e-15: return np.nan, np.nan, np.inf
    return p, -q/p, np.sum(w * (y - (p*v + q*u))**2)

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

# Profile (coarse grid, then refined)
print("H0 profile scan...")
H0_grid = np.concatenate([np.arange(55, 65.25, 0.5), np.arange(65, 70.25, 0.2), np.arange(70, 82.25, 0.5)])
C_grid = np.arange(-0.5, 3.0, 0.25)

joint_prof = np.full(len(H0_grid), np.nan)
hz_prof = np.full(len(H0_grid), np.nan)
best = (None, np.inf)

for i, H0 in enumerate(H0_grid):
    best_j = np.inf; best_h = np.inf
    best_abc = None
    for C in C_grid:
        A, B, chi2_h = fit_AB(H0, C)
        if not np.isfinite(chi2_h): continue
        if chi2_h < best_h: best_h = chi2_h
        chi2_s = chi2_sne(A, B, C, H0)
        if not np.isfinite(chi2_s): continue
        chi2_j = chi2_h + chi2_s
        if chi2_j < best_j:
            best_j = chi2_j
            best_abc = (A, B, C, chi2_h, chi2_s)
    joint_prof[i] = best_j
    hz_prof[i] = best_h
    if best_j < best[1] and best_abc is not None:
        best = ((H0, *best_abc), best_j)

joint_prof -= np.nanmin(joint_prof)
hz_prof -= np.nanmin(hz_prof)

H0_best, A_best, B_best, C_best, ch_best, cs_best = best[0]

def crossing(h, d, target):
    valid = np.isfinite(d)
    h, d = h[valid], d[valid]
    above = d >= target
    if np.sum(above) < 2: return None, None
    i = np.where(above[:-1] != above[1:])[0]
    if len(i) == 0: return None, None
    vals = [h[ix] + (target-d[ix])/(d[ix+1]-d[ix])*(h[ix+1]-h[ix]) for ix in i]
    return min(vals), max(vals)

lo1, hi1 = crossing(H0_grid, joint_prof, 1.0)
lo2, hi2 = crossing(H0_grid, joint_prof, 4.0)

print(f"\n{'='*60}")
print(f"  H0 = {H0_best:.2f} km/s/Mpc  (68% CL = [{lo1:.2f}, {hi1:.2f}])")
print(f"  95% CL = [{lo2:.2f}, {hi2:.2f}]")
print(f"  chi2_H = {ch_best:.1f}/32, chi2_SN = {cs_best:.1f}/1589")
print(f"  Planck 67.4±0.5: Δ = {H0_best-67.4:+.1f}σ = {(H0_best-67.4)/0.5:+.1f}")
print(f"  SH0ES 73.0±1.0:  Δ = {H0_best-73.0:+.1f}σ = {(H0_best-73.0)/1.0:+.1f}")
print(f"{'='*60}")

# Plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid = np.isfinite(joint_prof)
    ax.plot(H0_grid[valid], joint_prof[valid], 'b-', lw=2, label='Joint (CC+BAO+SNe)')
    valid_h = np.isfinite(hz_prof)
    ax.plot(H0_grid[valid_h], hz_prof[valid_h], 'r-', lw=1.5, alpha=0.6, label='CC+BAO only')
    
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.text(80.8, 1.0, '68% CL', va='bottom', fontsize=9, color='gray')
    ax.axhline(4.0, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.text(80.8, 4.0, '95% CL', va='bottom', fontsize=9, color='gray')
    
    if lo1 and hi1:
        ax.axvspan(lo1, hi1, alpha=0.12, color='blue', label=f'68% CL [{lo1:.1f}, {hi1:.1f}]')
    if lo2 and hi2:
        ax.axvspan(lo2, hi2, alpha=0.06, color='blue')
    
    for val, label, c in [(67.4, 'Planck', 'green'), (73.0, 'SH0ES', 'orange')]:
        ax.axvline(val, color=c, ls='-', lw=1.5, alpha=0.7)
        yl = ax.get_ylim()[1]
        ax.text(val, yl*0.92, label, rotation=90, fontsize=10, color=c, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('H0 (km/s/Mpc)', fontsize=12)
    ax.set_ylabel('Δχ²', fontsize=12)
    ax.set_title('H0 Profile Likelihood: SR + CC+BAO + Pantheon+', fontsize=13)
    ax.set_xlim(54, 83)
    ax.set_ylim(0, max(10, np.nanmax(joint_prof)*1.1))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15)
    
    # Annotation
    ax.text(0.98, 0.98, f'H0 = {H0_best:.1f} [{lo1:.1f}, {hi1:.1f}] (68%)\n'
                       f'χ²_H/32 = {ch_best:.1f}, χ²_S = {cs_best:.1f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/tmp/h0_profile_plot.png', dpi=150)
    print("Plot: /tmp/h0_profile_plot.png")
except Exception as e:
    print(f"Plot error: {e}")

#!/usr/bin/env python3
"""ΛCDM joint fit to CC+BAO+DESI+Pantheon+ for comparison with SR result.

H(z) = H0 * sqrt(Ωm*(1+z)^3 + (1-Ωm))
Scan Ωm, precompute I(z) for all SNe at once via cumulative integration.
"""

import numpy as np, sys, time
sys.path.insert(0, '.')
from joint_rank import load_data
from pantheon_cov import load_cov

C = 299792.458
CONST = 5 * np.log10(C) + 25

t0 = time.time()
print("Loading data...")
hz = load_data()
z_h, H_h, e_h = hz[:,0], hz[:,1], hz[:,2]
z_sn, mu_sn, Cinv, s = load_cov()
n_sn = len(z_sn)
print(f"  {len(hz)} H(z), {n_sn} SNe")

# Precompute sorted unique SN redshifts for batch integration
z_grid = np.linspace(0, 2.5, 5001)  # fine grid for 1D interpolation
z_sn_sorted = np.sort(z_sn)

def compute_A(Om):
    """A_i(Om) = 5*log10(I(z_i)) + 5*log10(1+z_i) + CONST using cumulative Simpson."""
    integrand = 1.0 / np.sqrt(Om*(1+z_grid)**3 + (1-Om))
    h = z_grid[1] - z_grid[0]
    # Cumulative trapezoidal
    I_cum = np.cumsum((integrand[:-1] + integrand[1:]) / 2 * h)
    I_cum = np.insert(I_cum, 0, 0.0)
    # Interpolate to SN redshifts
    I_sn = np.interp(z_sn, z_grid, I_cum)
    return 5.0 * np.log10(I_sn) + 5.0 * np.log10(1+z_sn) + CONST

def solve_hz(Om):
    """Solve H0 analytically from H(z) data."""
    f = np.sqrt(Om*(1+z_h)**3 + (1-Om))
    w = 1/e_h**2
    H0 = np.sum(w * f * H_h) / np.sum(w * f**2)
    chi2 = np.sum(w * (H_h - H0*f)**2)
    return chi2, H0

def chi2_sn_fast(A_vals):
    """SNe chi2 with free M_offset, precomputed A_i."""
    good = np.isfinite(A_vals) & np.isfinite(mu_sn)
    if np.sum(good) < 10: return None, None
    g = np.where(good)[0]
    r = mu_sn[g] - A_vals[g]
    Cg = Cinv[np.ix_(g, g)]
    ones = np.ones(len(g))
    a = Cg @ ones; b = Cg @ r
    M_eff = np.sum(b) / np.sum(a)
    chi2 = np.dot(r, b) - np.sum(b)**2 / np.sum(a)
    return chi2, M_eff

print("\nScanning Ωm [0.10, 0.55]...")
best = (None, np.inf)
for Om in np.linspace(0.10, 0.55, 30):
    A = compute_A(Om)
    chi2h, H0 = solve_hz(Om)
    r = chi2_sn_fast(A)
    if r[0] is None: continue
    chi2s, M_eff = r
    j = chi2h + chi2s
    if j < best[1]:
        best = ((Om, H0, M_eff, chi2h, chi2s), j)
        print(f"  Ωm={Om:.3f} H0={H0:.1f} χ²_H={chi2h:.1f} χ²_SN={chi2s:.1f} joint={j:.1f}")

Om_b, H0_b, M_eff_b, ch_b, cs_b = best[0]
print(f"\nBest: Ωm={Om_b:.3f} H0={H0_b:.2f} χ²={ch_b+cs_b:.1f}  ({ch_b:.1f}+{cs_b:.1f})")

# Refine
print("Refining...")
for Om in np.linspace(max(0.10, Om_b-0.08), min(0.55, Om_b+0.08), 20):
    A = compute_A(Om)
    chi2h, H0 = solve_hz(Om)
    r = chi2_sn_fast(A)
    if r[0] is None: continue
    chi2s, M_eff = r
    if chi2h+chi2s < best[1]:
        Om_b, H0_b, M_eff_b, ch_b, cs_b = Om, H0, M_eff, chi2h, chi2s
        best = ((Om, H0, M_eff, chi2h, chi2s), chi2h+chi2s)

print(f"  Final: Ωm={Om_b:.3f} H0={H0_b:.2f}")
print(f"  χ²_H = {ch_b:.1f}  χ²_SN = {cs_b:.1f}  joint = {ch_b+cs_b:.1f}")
print(f"  M = {M_eff_b + 5*np.log10(H0_b):.3f} mag")

# SR comparison
print(f"\n{'─'*60}")
print("SR (Cpx 13, Pantheon+ full cov):")
print("  H0 = 68.3  χ²_H = 25.3  χ²_SN = 1405.3  joint = 1430.6")
print(f"ΛCDM:          H0 = {H0_b:.1f}  χ²_H = {ch_b:.1f}  χ²_SN = {cs_b:.1f}  joint = {ch_b+cs_b:.1f}")
print(f"Δjoint (ΛCDM - SR) = {ch_b+cs_b-1430.6:+.1f}")

print(f"\n({time.time()-t0:.0f}s)")

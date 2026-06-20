#!/usr/bin/env python3
"""
Validate symbolic-regression H(z) models against Pantheon+ SNe Ia data.

For each candidate H(z) model (from hubble_pilot.py runs):
  1. Numerically integrate H(z) -> comoving distance -> luminosity distance
  2. Predict distance modulus mu(z) = 5 log10(D_L / Mpc) + 25
  3. Compare with Pantheon+ binned distance moduli
  4. Report chi^2 for each model

Usage:
  python3 pantheon_validate.py
"""

import numpy as np
import urllib.request
import sys
# Integration using simple adaptive quadrature (avoids scipy dependency)
def quad_simple(f, a, b, n=1000):
    """Simple adaptive Simpson's rule for single integrals."""
    xs = np.linspace(a, b, 2*n + 1)
    h = (b - a) / (2*n)
    fx = f(xs)
    return h/3 * (fx[0] + fx[-1] + 4*np.sum(fx[1::2]) + 2*np.sum(fx[2:-1:2]))

# ============================================================
# 1. CANDIDATE H(z) MODELS (from hubble_pilot.py runs)
# ============================================================

def H_model_cpx20(z):
    """Best-fit model (complexity 20, loss=101.7), H0=64.42"""
    H0_ref = 67.4
    f = -48.35652340908182 * z * (z - 2.778854840338059) \
        * np.sqrt(z * (z - 0.8499296350310853) + 0.39084941382533883) \
        - 2.978402579262088
    return H0_ref + f

def H_model_cpx11(z):
    """Complexity 11 model (loss=106.3), H0=72.10"""
    H0_ref = 67.4
    f = z * z * (84.35746021110187 + z * (-27.166125568007434)) + 4.7032
    return H0_ref + f

def H_model_lcdm_planck(z, H0=67.4, Om=0.315):
    """LCDM with Planck parameters."""
    Ol = 1.0 - Om
    return H0 * np.sqrt(Om * (1+z)**3 + Ol)

def H_model_lcdm_sh0es(z, H0=73.0, Om=0.315):
    """LCDM with SH0ES H0."""
    return H_model_lcdm_planck(z, H0, Om)

# ============================================================
# 2. INTEGRATION: H(z) -> mu(z)
# ============================================================

C_OVER_10 = 299792.458 / 10.0  # c in km/s, /10 for Mpc -> 10pc

def mu_from_H(H_func, z):
    """Compute distance modulus mu(z) from H(z) model.
    
    mu(z) = 5 log10( D_L(z) / 10pc )
    D_L(z) = (1+z) * c * int_0^z dz' / H(z')
    """
    # Comoving distance: Dc = c * int_0^z dz' / H(z')
    Dc = 299792.458 * quad_simple(lambda zp: 1.0 / H_func(zp), 0, z)
    
    # Luminosity distance (Mpc)
    DL = (1 + z) * Dc
    
    # Distance modulus
    mu = 5.0 * np.log10(DL) + 25.0
    return mu

# Vectorized version
def mu_array(H_func, z_arr):
    return np.array([mu_from_H(H_func, z) for z in z_arr])

# ============================================================
# 3. FETCH PANTHEON+ DATA
# ============================================================

PANTHEON_URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
    "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES.dat")

def fetch_pantheon():
    """Fetch Pantheon+ data, return (zHD, mu, mu_err) arrays.
    
    Columns: zHD(z), MU_SH0ES(10), MU_SH0ES_ERR_DIAG(11)
    Filters: zHD > 0.01, USED_IN_SH0ES_HF(14) = 1
    """
    print("  Fetching Pantheon+ data...")
    req = urllib.request.urlopen(PANTHEON_URL, timeout=30)
    content = req.read().decode()
    
    lines = content.strip().split('\n')
    header = lines[0].split()
    
    z_idx = header.index('zHD')
    mu_idx = header.index('MU_SH0ES')
    mu_err_idx = header.index('MU_SH0ES_ERR_DIAG')
    use_idx = header.index('USED_IN_SH0ES_HF')
    
    print(f"  Found columns: zHD={z_idx}, MU_SH0ES={mu_idx}, err={mu_err_idx}")
    
    z_list, mu_list, err_list = [], [], []
    for line in lines[1:]:
        cols = line.split()
        if len(cols) <= max(z_idx, mu_idx, mu_err_idx):
            continue
        try:
            z = float(cols[z_idx])
            mu = float(cols[mu_idx])
            mu_err = float(cols[mu_err_idx])
            used = int(cols[use_idx]) if len(cols) > use_idx else 1
        except (ValueError, IndexError):
            continue
        
        # Filter: z>0.01 to avoid local flow issues, and used in HF fit
        if z > 0.01 and mu_err > 0 and used >= 0:
            z_list.append(z)
            mu_list.append(mu)
            err_list.append(mu_err)
    
    return (np.array(z_list), np.array(mu_list), np.array(err_list))

# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Pantheon+ Validation of SR H(z) Models")
    print("=" * 60)
    
    # Fetch data
    z_p, mu_p, err_p = fetch_pantheon()
    print(f"  N_SNe = {len(z_p)}")
    print(f"  z range: [{z_p.min():.4f}, {z_p.max():.4f}]")
    
    # Candidate models
    models = [
        ("Cpx 20 (best-fit, H0=64.4)", H_model_cpx20),
        ("Cpx 11 (H0=72.1)",          H_model_cpx11),
        ("LCDM Planck (H0=67.4)",     H_model_lcdm_planck),
        ("LCDM SH0ES (H0=73.0)",      H_model_lcdm_sh0es),
    ]
    
    print(f"\n  {'Model':>30} {'chi2_fixed':>10} {'chi2_freeM':>10} {'chi2/dof':>8} {'H0':>7} {'dM':>7}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*7}")
    
    for name, hfunc in models:
        mu_pred = mu_array(hfunc, z_p)
        resid = mu_p - mu_pred
        good = np.isfinite(mu_pred) & np.isfinite(resid)
        
        # Chi2 with fixed normalization
        chi2_fixed = np.sum((resid[good] / err_p[good])**2)
        
        # Chi2 marginalizing over absolute magnitude M (additive constant)
        w = 1.0 / err_p[good]**2
        delta_m = np.sum(resid[good] * w) / np.sum(w)  # optimal offset
        resid_free = resid[good] - delta_m
        chi2_free = np.sum((resid_free / err_p[good])**2)
        
        ndof = np.sum(good)
        H0_val = hfunc(np.array([0.0]))[0]
        
        print(f"  {name:>30} {chi2_fixed:>10.1f} {chi2_free:>10.1f} {chi2_free/ndof:>8.3f} {H0_val:>7.1f} {delta_m:>7.2f}")
    
    print("\nDone.")

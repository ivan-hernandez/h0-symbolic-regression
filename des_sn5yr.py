#!/usr/bin/env python3
"""DES-SN5YR data loader with full covariance."""

import numpy as np
import urllib.request, os, sys
import joint_rank as jr

HD_URL = ("https://raw.githubusercontent.com/des-science/DES-SN5YR/main/"
    "4_DISTANCES_COVMAT/DES-Dovekie_HD.csv")
COV_URL = ("https://github.com/des-science/DES-SN5YR/raw/main/"
    "4_DISTANCES_COVMAT/STAT%2BSYS.npz")

CACHE = "/tmp/DES_SN5YR_cache.npz"

def load(use_cache=True):
    if use_cache and os.path.exists(CACHE):
        d = np.load(CACHE)
        return d['z'], d['mu'], d['Cinv'], float(d['Cinv_1sum'])
    t0 = __import__('time').time()
    # Data
    print("  Reading DES-SN5YR data...")
    req = urllib.request.urlopen(HD_URL, timeout=30)
    zl, mul = [], []
    for line in req.read().decode().strip().split('\n'):
        if not line.startswith('SN:'): continue
        cols = line.split()
        try:
            z = float(cols[3]); mu = float(cols[5])
        except: continue
        if z > 0.01:
            zl.append(z); mul.append(mu)
    z, mu = np.array(zl), np.array(mul)
    n = len(z)
    print(f"  {n} SNe (z>0.01)")
    # Covariance
    print("  Loading covariance...")
    req2 = urllib.request.urlopen(COV_URL, timeout=60)
    # The URL already has the raw file, so we can save and load
    # Actually, COV_URL already points to the raw file, but GitHub's raw
    # content URL for binary files works differently.
    pass  # Download below if not cached

def load_cached():
    """Alternative: load pre-downloaded files."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d['z'], d['mu'], d['Cinv'], float(d['Cinv_1sum'])
    t0 = __import__('time').time()
    # Download HD data
    print("  Reading DES-SN5YR HD data...")
    req = urllib.request.urlopen(HD_URL, timeout=30)
    zl, mul = [], []
    for line in req.read().decode().strip().split('\n'):
        if not line.startswith('SN:'): continue
        cols = line.split()
        try: z = float(cols[3]); mu = float(cols[5])
        except: continue
        if z > 0.01: zl.append(z); mul.append(mu)
    z = np.array(zl); mu = np.array(mul)
    n = len(z)
    print(f"  {n} SNe (z>0.01)")
    # Download covariance (may need separate request for binary)
    cov_file = "/tmp/DES_cov_raw.npz"
    if not os.path.exists(cov_file):
        print("  Downloading DES covariance (6 MB)...")
        resp = urllib.request.urlopen(COV_URL, timeout=120)
        with open(cov_file, 'wb') as f: f.write(resp.read())
    cov_data = np.load(cov_file)
    nsn = int(cov_data['nsn'][0])
    print(f"  Inverse covariance: {nsn} SNe (file stores inverse, not covariance)")
    flat = cov_data['cov']  # upper triangle of inverse covariance
    # Unpack upper triangular → full symmetric matrix
    Cinv = np.zeros((nsn, nsn), dtype=np.float64)
    Cinv[np.triu_indices(nsn)] = flat
    Cinv += Cinv.T
    np.fill_diagonal(Cinv, np.diag(Cinv) / 2)  # diagonal was doubled
    # Verify symmetry
    assert np.allclose(Cinv, Cinv.T), "Precision matrix not symmetric!"
    print(f"  Using precision matrix directly ({nsn}x{nsn})")
    ones = np.ones(nsn)
    Cinv_1sum = float(np.sum(Cinv @ ones))
    np.savez(CACHE, z=z, mu=mu, Cinv=Cinv, Cinv_1sum=Cinv_1sum)
    t = __import__('time').time() - t0
    print(f"  Cached to {CACHE} ({t:.0f}s)")
    return z, mu, Cinv, Cinv_1sum

def chi2_sn_cov(H0, A, B, C, z_sn, mu_sn, Cinv, Cinv_1sum):
    """SNe chi2 with full covariance, marginalised over M."""
    from joint_rank import mu_from_H
    def Hf(z): return H0 + A*z*(z-B)*(z**2+C)
    mu_pred = np.array([mu_from_H(Hf, z) for z in z_sn])
    r0 = mu_sn - mu_pred
    good = np.isfinite(mu_pred) & np.isfinite(mu_sn)
    if np.sum(good) < 10: return None
    g = np.where(good)[0]
    r = r0[g]; Cg = Cinv[np.ix_(g, g)]
    ones = np.ones(len(g)); a = Cg @ ones; denom = np.sum(a)
    b = Cg @ r; dm = np.sum(b) / denom
    chi2 = np.dot(r, b) - (np.sum(b))**2 / denom
    return chi2, dm

if __name__ == "__main__":
    z, mu, Cinv, s = load_cached()
    print(f"  z=[{z.min():.3f},{z.max():.3f}], {len(z)} SNe")
    print(f"  mu=[{mu.min():.3f},{mu.max():.3f}]")
    print(f"  1^T C^-1 1 = {s:.2f}")

#!/usr/bin/env python3
"""Pantheon+ data loader with full systematic covariance matrix.

Usage:
    from pantheon_cov import load_cov
    z, mu, Cinv, Cinv_1sum = load_cov()
"""

import numpy as np
import urllib.request
import sys, os, time, pickle

DATA_URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
    "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES.dat")
COV_URL = ("https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES_STAT%2BSYS.cov")
CACHE = "/tmp/pantheon_cov_cache.npz"

def _download_cov():
    """Download covariance file if not cached."""
    fname = "/tmp/pantheon_cov.txt"
    if not os.path.exists(fname):
        print("  Downloading covariance matrix (32 MB)...")
        req = urllib.request.urlopen(COV_URL, timeout=120)
        with open(fname, 'wb') as f:
            f.write(req.read())
    return fname

def load_cov(use_cache=True):
    """Load Pantheon+ distance moduli and full covariance.

    Returns:
        z: array of redshifts (n,)
        mu: array of MU_SH0ES (n,)
        Cinv: inverse covariance matrix (n x n)
        Cinv_1sum: scalar = 1^T C^{-1} 1
    """
    if use_cache and os.path.exists(CACHE):
        print("  Loading cached covariance...")
        d = np.load(CACHE)
        return d['z'], d['mu'], d['Cinv'], float(d['Cinv_1sum'])

    t0 = time.time()
    cov_file = _download_cov()

    # Data file
    print("  Reading data file...")
    req = urllib.request.urlopen(DATA_URL, timeout=30)
    content = req.read().decode()
    lines = content.strip().split('\n')
    header = lines[0].split()
    zi = header.index('zHD'); mi = header.index('MU_SH0ES')
    ei = header.index('MU_SH0ES_ERR_DIAG')
    ui = header.index('USED_IN_SH0ES_HF')

    rows = []
    for line in lines[1:]:
        cols = line.split()
        if len(cols) <= ui: continue
        try:
            z = float(cols[zi]); m = float(cols[mi])
            e = float(cols[ei]); used = int(cols[ui])
        except: continue
        rows.append((z, m, e, used))
    n_all = len(rows)

    # Covariance
    print("  Reading covariance matrix...")
    with open(cov_file) as f:
        cov_lines = f.readlines()
    n_cov = int(cov_lines[0].strip())
    assert n_cov == n_all, f"Size: cov={n_cov} data={n_all}"

    keep = [i for i in range(n_all) if rows[i][0] > 0.01 and rows[i][3] >= 0]
    n_keep = len(keep)
    print(f"  {n_keep} SNe after filtering")

    z = np.array([rows[i][0] for i in keep])
    mu = np.array([rows[i][1] for i in keep])

    # Subset
    cov_flat = np.array([float(x) for x in cov_lines[1:]])
    C = cov_flat.reshape((n_cov, n_cov))
    C_sub = C[np.ix_(keep, keep)]

    # Invert
    print(f"  Inverting {n_keep}x{n_keep} covariance...")
    Cinv = np.linalg.inv(C_sub)

    ones = np.ones(n_keep)
    Cinv_1sum = float(np.sum(Cinv @ ones))

    np.savez(CACHE, z=z, mu=mu, Cinv=Cinv, Cinv_1sum=Cinv_1sum)
    print(f"  Cached to {CACHE}")
    t = time.time() - t0
    print(f"  Done ({t:.0f}s)")
    return z, mu, Cinv, Cinv_1sum

def chi2_sn_cov(H0, A, B, C, z_sn, mu_sn, Cinv, Cinv_1sum):
    """SNe chi2 with full covariance, marginalised over M.

    Returns (chi2, delta_M) where delta_M is the best-fit M offset.
    """
    from joint_rank import mu_from_H
    def Hf(z): return H0 + A*z*(z-B)*(z**2+C)
    mu_pred = np.array([mu_from_H(Hf, z) for z in z_sn])
    r0 = mu_sn - mu_pred
    good = np.isfinite(mu_pred) & np.isfinite(mu_sn)
    if np.sum(good) < 10: return None
    g = np.where(good)[0]
    r = r0[g]
    Cg = Cinv[np.ix_(g, g)]
    ones = np.ones(len(g))
    a = Cg @ ones  # C^{-1} * 1
    denom = np.sum(a)
    b = Cg @ r     # C^{-1} * r
    dm = np.sum(b) / denom
    chi2 = np.dot(r, b) - (np.sum(b))**2 / denom
    return chi2, dm

if __name__ == "__main__":
    z, mu, Cinv, s = load_cov(use_cache=False)
    print(f"  z=[{z.min():.3f},{z.max():.3f}], {len(z)} SNe")
    print(f"  mu=[{mu.min():.3f},{mu.max():.3f}]")
    print(f"  1^T C^-1 1 = {s:.2f}")

    # Quick test: baseline fit
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from joint_rank import load_data, mu_from_H
    hz = load_data()
    z_h, H, e = hz[:,0], hz[:,1], hz[:,2]

    print(f"\n  Baseline (Cpx 13, full cov)...")
    best = (None, np.inf)
    for C in np.linspace(-0.5, 3.0, 50):
        u = z_h*(z_h**2+C); v = z_h**2*(z_h**2+C)
        X = np.column_stack([np.ones_like(z_h), v, u])
        w = 1.0/e**2
        try: beta = np.linalg.solve(X.T@(X*w[:,None]), X.T@(w*H))
        except: continue
        H0, p, q = beta
        if abs(p) < 1e-15: continue
        A, B = p, -q/p
        chi2_h = np.nansum(w*(H-(H0+p*v+q*u))**2)
        r = chi2_sn_cov(H0, A, B, C, z, mu, Cinv, s)
        if r is None: continue
        chi2_s, dm = r
        j = chi2_h + chi2_s
        if j < best[1]: best = ((H0, A, B, C, chi2_h, chi2_s, dm), j)
    if best[0]:
        H0, A, B, C, ch, cs, dm = best[0]
        print(f"  H0={H0:.2f} A={A:.2f} B={B:.2f} C={C:.2f}")
        print(f"  chi2_H={ch:.1f} chi2_SN(cov)={cs:.1f} dM={dm:.3f}")
        # Diagonal for comparison
        from joint_rank import fetch_pantheon
        zd, md, ed = fetch_pantheon()
        def Hf(z): return H0 + A*z*(z-B)*(z**2+C)
        mp = np.array([mu_from_H(Hf, z) for z in zd])
        resid = md - mp; g = np.isfinite(mp)&np.isfinite(md)
        ws = 1.0/ed[g]**2; dm_d = np.sum(resid[g]*ws)/np.sum(ws)
        cs_d = np.sum(((resid[g]-dm_d)/ed[g])**2)
        print(f"  chi2_SN(diag)={cs_d:.1f} dM(diag)={dm_d:.3f}")

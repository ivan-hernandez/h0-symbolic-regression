#!/usr/bin/env python3
"""
Joint evaluation of SR-discovered H(z) models on CC+BAO+SNe data.

Reads hall_of_fame.csv from a PySR run, evaluates each expression
on both H(z) data (CC+BAO) and distance modulus data (Pantheon+),
and ranks by combined merit.
"""

import numpy as np
import csv, os, sys, urllib.request

# ---------- data loaders (copied from hubble_pilot.py) ----------

def get_cc_data():
    cc = np.array([
        [0.070,    69.0,     19.6],
        [0.090,    69.0,     12.0],
        [0.120,    68.6,     26.2],
        [0.170,    83.0,      8.0],
        [0.1791,   75.0,      4.0],
        [0.1993,   75.0,      5.0],
        [0.200,    72.9,     29.6],
        [0.270,    77.0,     14.0],
        [0.280,    88.8,     36.6],
        [0.3519,   83.0,     14.0],
        [0.3802,   83.0,     13.5],
        [0.400,    95.0,     17.0],
        [0.4004,   77.0,     10.2],
        [0.4247,   87.1,     11.2],
        [0.4497,   92.8,     12.9],
        [0.470,    89.0,     34.0],
        [0.4783,   80.9,      9.0],
        [0.480,    97.0,     62.0],
        [0.5929,  104.0,     13.0],
        [0.6797,   92.0,      8.0],
        [0.750,    98.8,     33.6],
        [0.7812,  105.0,     12.0],
        [0.800,   113.1,     28.5],
        [0.8754,  125.0,     17.0],
        [0.880,    90.0,     40.0],
        [0.900,   117.0,     23.0],
        [1.037,   154.0,     20.0],
        [1.300,   168.0,     17.0],
        [1.363,   160.0,     33.6],
        [1.430,   177.0,     18.0],
        [1.530,   140.0,     14.0],
        [1.750,   202.0,     40.0],
        [1.965,   186.5,     50.4],
    ])
    return cc

def get_bao_data():
    bao = np.array([
        [0.380,    81.1,      2.2],
        [0.510,    91.1,      2.1],
        [0.610,    99.4,      2.2],
    ])
    return bao

def get_desi_bao_data(r_d=147.0):
    """
    DESI DR1 BAO: D_H/r_d values from Table 1 of DESI 2024 VI (arXiv:2404.03002).
    Converted to H(z) = c / (r_d * D_H/r_d) with r_d = {r_d} Mpc.
    """
    c = 299792.458
    desi = np.array([
        # z, D_H/r_d, err
        [0.510, 20.98334647, 0.61],
        [0.706, 20.07872919, 0.60],
        [0.930, 17.87612922, 0.35],
        [1.317, 13.82372285, 0.42],
        [2.330,  8.52256583, 0.17],
    ])
    hz = c / (r_d * desi[:, 1])
    errs = hz * desi[:, 2] / desi[:, 1]
    return np.column_stack([desi[:, 0], hz, errs])

def load_data(include_desi=True):
    cc = get_cc_data()
    bao = get_bao_data()
    combined = np.vstack([cc, bao])
    if include_desi:
        desi = get_desi_bao_data()
        combined = np.vstack([combined, desi])
    combined = combined[combined[:, 0].argsort()]
    mask = (combined[:, 2] > 0) & (combined[:, 1] > 0) & (combined[:, 2] < 100)
    return combined[mask]

# ---------- Pantheon+ data ----------

PANTHEON_URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
    "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    "Pantheon%2BSH0ES.dat")

def fetch_pantheon():
    req = urllib.request.urlopen(PANTHEON_URL, timeout=30)
    lines = req.read().decode().strip().split('\n')
    hdr = lines[0].split()
    zi = hdr.index('zHD')
    mi = hdr.index('MU_SH0ES')
    ei = hdr.index('MU_SH0ES_ERR_DIAG')
    z, m, e = [], [], []
    for line in lines[1:]:
        c = line.split()
        if len(c) <= max(zi, mi, ei): continue
        zh, mu, err = float(c[zi]), float(c[mi]), float(c[ei])
        if zh > 0.01 and err > 0:
            z.append(zh); m.append(mu); e.append(err)
    return np.array(z), np.array(m), np.array(e)

# ---------- integration ----------

def quad_simple(f, a, b, n=2000):
    xs = np.linspace(a, b, 2*n + 1)
    h = (b - a) / (2*n)
    fx = f(xs)
    return h/3 * (fx[0] + fx[-1] + 4*np.sum(fx[1::2]) + 2*np.sum(fx[2:-1:2]))

def mu_from_H(H_func, z):
    Dc = 299792.458 * quad_simple(lambda zp: 1.0 / H_func(zp), 0, z)
    DL = (1 + z) * Dc
    return 5.0 * np.log10(DL) + 25.0

# ---------- expression evaluation ----------

def make_H_func(eq_str):
    """Convert Julia expression string to Python H(z) function with free H0.
    
    The equation is f(z) = H(z) - H0_ref, so H(z) = H0_ref + f(z).
    We use H0_ref = 67.4 as in the SR run.
    """
    H0_ref = 67.4
    
    # Clean Julia syntax for Python eval
    cleaned = eq_str.strip()
    # Remove leading/trailing quotes
    if cleaned.startswith('"') and cleaned.endswith('"'): cleaned = cleaned[1:-1]
    # The expressions use sqrt(), *, +, - only — no ^ operator in this run
    # (If ^ appears in future runs, replace ^( with **()
    
    # Create eval environment
    env = {
        'x0': None,  # placeholder
        'sqrt': np.sqrt,
        'abs': np.abs,
        'exp': np.exp,
        'log': np.log,
    }
    
    def H_func(z_arr):
        env['x0'] = z_arr
        f = eval(cleaned, {"__builtins__":{}}, env)
        return H0_ref + np.asarray(f, dtype=float)
    
    return H_func

# ---------- main ----------

if __name__ == "__main__":
    hof_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hof.csv"
    
    print("Loading data...")
    data_hz = load_data()
    z_h, H_h, e_h = data_hz[:, 0], data_hz[:, 1], data_hz[:, 2]
    
    z_sn, mu_sn, e_sn = fetch_pantheon()
    print(f"  CC+BAO: {len(z_h)} points, Pantheon+: {len(z_sn)} points")
    
    # Read HoF
    equations = []
    with open(hof_file) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            cpx = int(row[0])
            loss = float(row[1])
            eq = row[2]
            equations.append((cpx, loss, eq))
    
    print(f"\n  {'#':>3} {'Cpx':>4} {'Loss_H':>8} {'chi2_H':>8} {'chi2_mu':>8} {'chi2_j':>8}   Equation (truncated)")
    print(f"  {'---':>3} {'----':>4} {'------':>8} {'-------':>8} {'-------':>8} {'-------':>8}   {'-'*35}")
    
    results = []
    for idx, (cpx, loss_h, eq) in enumerate(equations):
        try:
            H_func = make_H_func(eq)
        except Exception as exc:
            print(f"  {idx:>3} {cpx:>4} {loss_h:>8.1f} PARSE_ERR({type(exc).__name__}): {exc}")
            continue
        
        # chi2 on H(z) data
        try:
            H_pred = H_func(z_h)
            chi2_h = np.nansum(((H_h - H_pred) / e_h)**2)
        except Exception as exc:
            print(f"  {idx:>3} {cpx:>4} {loss_h:>8.1f} EVAL_ERR({type(exc).__name__}): {exc}")
            continue
        
        # chi2 on SNe (with free M)
        chi2_mu = np.nan
        try:
            mu_pred = np.array([mu_from_H(H_func, z) for z in z_sn])
            resid = mu_sn - mu_pred
            good = np.isfinite(mu_pred) & np.isfinite(resid)
            if np.sum(good) > 10:
                w = 1.0 / e_sn[good]**2
                delta_m = np.sum(resid[good] * w) / np.sum(w)
                chi2_mu = np.sum(((resid[good] - delta_m) / e_sn[good])**2)
        except Exception as exc:
            chi2_mu = np.nan
        
        chi2_j = chi2_h + (chi2_mu if not np.isnan(chi2_mu) else 0)
        
        eq_short = eq[:50]
        print(f"  {idx:>3} {cpx:>4} {loss_h:>8.1f} {chi2_h:>8.1f} {chi2_mu:>8.1f} {chi2_j:>8.1f}   {eq_short}")
        
        results.append((chi2_j, chi2_h, chi2_mu, cpx, idx, eq_short))
    
    results.sort()
    print(f"\n  Ranked by joint chi2:")
    print(f"  {'Rank':>4} {'Cpx':>4} {'chi2_H':>8} {'chi2_mu':>8} {'chi2_j':>8}   Equation")
    print(f"  {'-'*4} {'-'*4} {'-'*8} {'-'*8} {'-'*8}   {'-'*35}")
    for rank, (cj, ch, cm, cpx, _, eq) in enumerate(results):
        print(f"  {rank+1:>4} {cpx:>4} {ch:>8.1f} {cm:>8.1f} {cj:>8.1f}   {eq}")
    
    print("\nDone.")

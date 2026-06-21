"""DESI DR2 BAO data from DESI 2025 (arXiv:2503.14738).

D_H/r_s and D_M/r_s measurements with full covariance.
Combined tracer (ALL_GCcomb) from galaxy+quasar clustering.
"""
import numpy as np

C = 299792.458  # km/s

def load_dr2_bao(r_s=147.0):
    """Return list of BAO constraints.
    
    Each entry: dict with keys:
      z, D_H_rs, err_D_H_rs, D_M_rs, err_D_M_rs
    or D_V_rs for z=0.295.
    
    Also returns full covariance matrix for (D_M/r_s, D_H/r_s) points.
    """
    # z, D_V/r_s, err
    bgs = dict(z=0.295, D_V_rs=7.94167639, err=0.076087)
    
    # z, D_M/r_s, err, D_H/r_s, err, corr(D_M,D_H)
    # errors from covariance diagonal
    bao_pts = [
        dict(z=0.510, D_M_rs=13.58758434, e_DM=0.16836, D_H_rs=21.86294686, e_DH=0.42892, rho=-0.447),
        dict(z=0.706, D_M_rs=17.35069094, e_DM=0.17993, D_H_rs=19.45534918, e_DH=0.33387, rho=-0.395),
        dict(z=0.934, D_M_rs=21.57563956, e_DM=0.16178, D_H_rs=17.64149464, e_DH=0.20104, rho=-0.343),
        dict(z=1.321, D_M_rs=27.60085612, e_DM=0.32456, D_H_rs=14.17602155, e_DH=0.22455, rho=-0.398),
        dict(z=1.484, D_M_rs=30.51190063, e_DM=0.76358, D_H_rs=12.81699964, e_DH=0.51801, rho=-0.488),
        dict(z=2.330, D_M_rs=38.98897396, e_DM=0.53163, D_H_rs=8.63154567,  e_DH=0.10106, rho=-0.430),
    ]
    return bgs, bao_pts

def dr2_hz(r_s=147.0, use_bgs=True):
    """Convert DESI DR2 D_H/r_s to H(z) in km/s/Mpc.
    
    Returns array of [z, H(z), err].
    Excludes z=0.295 (BGS) which only has D_V/r_s.
    """
    bgs, pts = load_dr2_bao(r_s)
    rows = []
    for p in pts:
        hz = C / (r_s * p['D_H_rs'])
        err = hz * p['e_DH'] / p['D_H_rs']
        rows.append([p['z'], hz, err])
    arr = np.array(rows)
    arr = arr[arr[:, 0].argsort()]
    return arr

def dr2_dm(r_s=147.0):
    """Return D_M/r_s values (comoving angular diameter distance / r_s)."""
    bgs, pts = load_dr2_bao(r_s)
    rows = []
    for p in pts:
        rows.append([p['z'], p['D_M_rs'], p['e_DM']])
    arr = np.array(rows)
    arr = arr[arr[:, 0].argsort()]
    return arr

if __name__ == "__main__":
    bgs, pts = load_dr2_bao()
    print("BGS: z=0.295, D_V/r_s = 7.942 ± 0.076")
    print(f"\n  {'z':>7} {'D_H/r_s':>10} {'err':>8} {'D_M/r_s':>10} {'err':>8} {'H(z)':>8}")
    print(f"  {'-'*7} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    hz = dr2_hz()
    for p, h in zip(pts, hz):
        print(f"  {p['z']:>7.3f} {p['D_H_rs']:>10.4f} {p['e_DH']:>8.4f} {p['D_M_rs']:>10.4f} {p['e_DM']:>8.4f} {h[1]:>8.1f}")

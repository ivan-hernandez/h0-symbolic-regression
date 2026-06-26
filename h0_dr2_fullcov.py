#!/usr/bin/env python3
"""Cpx 13 full covariance fit with DESI DR2 — baseline + fix-M test.
Fix-M refits ALL parameters (H0,A,B,C) with M fixed to SH0ES value."""
import numpy as np, sys, time
sys.path.insert(0, '.')
from data import load_hz, mu_from_H
from pantheon_cov import load_cov as load_pp, chi2_sn_cov as chi2_pp
from des_sn5yr import load_cached as load_des, chi2_sn_cov as chi2_des

t0 = time.time()
hz = load_hz(version='dr2')
z_h, H_h, e_h = hz[:,0], hz[:,1], hz[:,2]
print(f"DR2 H(z): {len(hz)} pts")

def fit_generic(z, H, e, chi2_fn):
    """Scan C, solve (H0,A,B) analytically for given chi2_fn(H0,A,B,C)."""
    best = (None, np.inf)
    for C in np.linspace(-0.5, 3.0, 100):
        u = z*(z**2+C); v = z**2*(z**2+C)
        X = np.column_stack([np.ones_like(z), v, u]); w = 1.0/e**2
        try: beta = np.linalg.solve(X.T@(X*w[:,None]), X.T@(w*H))
        except: continue
        H0, p, q = beta
        if abs(p) < 1e-15: continue
        A, B = p, -q/p
        ch = np.nansum(w*(H-(H0+p*v+q*u))**2)
        r = chi2_fn(H0, A, B, C)
        if r is None: continue
        cs, extra = r  # extra is M_eff for free, 0 for fixed
        j = ch + cs
        if j < best[1]:
            best = ((H0, A, B, C, ch, cs, extra), j)
    return best

for name, load_sn, chi2fn in [("Pantheon+", load_pp, chi2_pp), ("DES-SN5YR", load_des, chi2_des)]:
    z_sn, mu_sn, Cinv, s = load_sn()
    print(f"\n{'='*60}\n  {name} full cov + DR2\n{'='*60}")

    # Baseline (free M)
    wrap = lambda H0,A,B,C, zs=z_sn,ms=mu_sn,Ci=Cinv,ss=s: chi2fn(H0,A,B,C,zs,ms,Ci,ss)
    best, _ = fit_generic(z_h, H_h, e_h, wrap)
    if best[0] is None: print("  FAILED"); continue
    H0, A, B, C, ch, cs, dm = best
    print(f"  BASELINE (free M): H0={H0:.2f}  A={A:.2f}  B={B:.2f}  C={C:.2f}")
    print(f"    chi2_H={ch:.1f}  chi2_SN={cs:.0f}  joint={ch+cs:.0f}  M_off={dm:.3f}")

    # Fix-M test: refit all params with M fixed to SH0ES (data already calibrated)
    def chi2_fixed(H0, A, B, C):
        def Hf(z): return H0 + A*z*(z-B)*(z**2+C)
        mp = np.array([mu_from_H(Hf, z) for z in z_sn])
        r0 = mu_sn - mp
        g = np.isfinite(mp) & np.isfinite(mu_sn)
        if np.sum(g) < 10: return None
        gg = np.where(g)[0]; Cg = Cinv[np.ix_(gg, gg)]
        return float(r0[gg] @ (Cg @ r0[gg])), 0.0

    best_fix, _ = fit_generic(z_h, H_h, e_h, chi2_fixed)
    if best_fix[0] is not None:
        H0_f, A_f, B_f, C_f, ch_f, cs_f, _ = best_fix
        dcs = cs_f - cs
        print(f"  FIX-M (SH0ES): H0={H0_f:.2f}  A={A_f:.2f}  B={B_f:.2f}  C={C_f:.2f}")
        print(f"    chi2_H={ch_f:.1f}  chi2_SN={cs_f:.0f}  joint={ch_f+cs_f:.0f}")
        print(f"    Delta chi2_SN = {dcs:.0f}  ({np.sqrt(max(dcs,0)):.1f}sigma)")

print(f"\n({time.time()-t0:.0f}s)")

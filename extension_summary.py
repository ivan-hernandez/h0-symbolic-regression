"""Final summary of all extensions for the AGENTS.md."""
import numpy as np
import sys, time
sys.path.insert(0, '.')
from data import load_hz, get_desi_dr1, get_desi_dr2
from pantheon_cov import load_cov, chi2_sn_cov

t0 = time.time()

# ── DR1 vs DR2 comparison ──
hz_dr1 = load_hz(version='dr1')
hz_dr2 = load_hz(version='dr2')

def fit_hz_only(hz):
    z, H, e = hz[:,0], hz[:,1], hz[:,2]
    best = (None, np.inf)
    for C in np.linspace(-0.5, 3.0, 50):
        u = z*(z**2+C); v = z**2*(z**2+C)
        X = np.column_stack([np.ones_like(z), v, u])
        w = 1/e**2
        try: beta = np.linalg.solve(X.T@(X*w[:,None]), X.T@(w*H))
        except: continue
        H0, p, q = beta
        if abs(p) < 1e-15: continue
        A, B = p, -q/p
        ch = np.nansum(w*(H-(H0+p*v+q*u))**2)
        if ch < best[1]: best = ((H0, A, B, C, ch), ch)
    return best[0]

def fit_joint(hz, chi2_fn, nC=40):
    z, H, e = hz[:,0], hz[:,1], hz[:,2]
    best = (None, np.inf)
    for C in np.linspace(-0.5, 3.0, nC):
        u = z*(z**2+C); v = z**2*(z**2+C)
        X = np.column_stack([np.ones_like(z), v, u])
        w = 1/e**2
        try: beta = np.linalg.solve(X.T@(X*w[:,None]), X.T@(w*H))
        except: continue
        H0, p, q = beta
        if abs(p) < 1e-15: continue
        A, B = p, -q/p
        ch = np.nansum(w*(H-(H0+p*v+q*u))**2)
        r = chi2_fn(H0, A, B, C)
        if r is None: continue
        cs, dm = r
        j = ch + cs
        if j < best[1]: best = ((H0, A, B, C, ch, cs, dm), j)
    return best[0]

print("=" * 70)
print("  EXTENSION RESULTS — FINAL SUMMARY")
print("=" * 70)

# 1. DR1 vs DR2 H(z) only
r1 = fit_hz_only(hz_dr1)
r2 = fit_hz_only(hz_dr2)
print(f"\n  ── H(z) ONLY ──")
print(f"  DR1 ({len(hz_dr1)} pts): H0={r1[0]:.1f}  A={r1[1]:.2f}  B={r1[2]:.2f}  C={r1[3]:.2f}  χ²={r1[4]:.1f}")
print(f"  DR2 ({len(hz_dr2)} pts): H0={r2[0]:.1f}  A={r2[1]:.2f}  B={r2[2]:.2f}  C={r2[3]:.2f}  χ²={r2[4]:.1f}")

# 2. DR1 vs DR2 joint
z_sn, mu_sn, Cinv, s = load_cov()
chi2_fn = lambda H0,A,B,C: chi2_sn_cov(H0,A,B,C,z_sn,mu_sn,Cinv,s)

r1j = fit_joint(hz_dr1, chi2_fn)
r2j = fit_joint(hz_dr2, chi2_fn)
print(f"\n  ── JOINT (H(z) + Pantheon+ full cov) ──")
if r1j: print(f"  DR1: H0={r1j[0]:.1f}  χ²_H={r1j[4]:.1f}  χ²_SN={r1j[5]:.0f}  joint={r1j[4]+r1j[5]:.1f}")
if r2j: print(f"  DR2: H0={r2j[0]:.1f}  χ²_H={r2j[4]:.1f}  χ²_SN={r2j[5]:.0f}  joint={r2j[4]+r2j[5]:.1f}")

# 3. Fix M test with DR2
def chi2_fixed(H0, A, B, C):
    from data import mu_from_H
    def Hf(z): return H0 + A*z*(z-B)*(z**2+C)
    mp = np.array([mu_from_H(Hf, z) for z in z_sn])
    r = mu_sn - mp
    good = np.isfinite(mp)&np.isfinite(mu_sn)
    if np.sum(good) < 10: return None
    g = np.where(good)[0]; Cg = Cinv[np.ix_(g, g)]
    chi2 = float(r[g] @ (Cg @ r[g]))
    return chi2, 0.0

r_fix = fit_joint(hz_dr2, chi2_fixed)
print(f"\n  ── FIX M (SH0ES calibration) ──")
if r_fix and r2j:
    dchi2 = (r_fix[4]+r_fix[5]) - (r2j[4]+r2j[5])
    print(f"  DR2: H0={r_fix[0]:.1f}  χ²_H={r_fix[4]:.1f}  χ²_SN={r_fix[5]:.0f}")
    print(f"  Δχ² vs free M = +{dchi2:.0f}")
    print(f"  Rejection: >{np.sqrt(dchi2):.1f}σ")

# 4. DESI DR1 vs DR2 direct comparison
pts_dr1 = get_desi_dr1()
pts_dr2 = get_desi_dr2()
print(f"\n  ── DESI DR1 vs DR2 H(z) comparison (r_d=147 Mpc) ──")
print(f"  {'z':>6} {'DR1 H':>8} {'err':>6} {'DR2 H':>8} {'err':>6} {'Δ/σ':>6}")
for d1 in pts_dr1:
    match = pts_dr2[np.abs(pts_dr2[:,0]-d1[0])<0.01]
    if len(match)==0: continue
    d2 = match[0]
    dz = d2[1]-d1[1]; sig = np.sqrt(d1[2]**2+d2[2]**2)
    print(f"  {d1[0]:>6.3f} {d1[1]:>8.1f} {d1[2]:>6.1f} {d2[1]:>8.1f} {d2[2]:>6.1f} {dz/sig:>+6.2f}")

# 5. External constraints
print(f"\n  ── EXTERNAL CONSTRAINTS ──")
ext = [
    ("GW170817 (VLBI, Gourdji+2026)", 65.5, 4.4),
    ("DES Y3+GW (Andrade-Oliveira+2026)", 67.94, 4.37),
    ("TDCOSMO 2025 (8 lenses)", 71.6, 3.6),
]
for name, h, e in ext:
    print(f"  {name:40s}  H0 = {h:.1f} ± {e:.1f}")
vals = np.array([65.5, 67.94, 71.6])
errs = np.array([4.4, 4.37, 3.6])
w = 1/errs**2
h_c = np.sum(vals*w)/np.sum(w)
e_c = 1/np.sqrt(np.sum(w))
print(f"  {'Combined external':40s}  H0 = {h_c:.1f} ± {e_c:.1f}")

# 6. Roman forecast
print(f"\n  ── ROMAN SPACE TELESCOPE LAUNCH (May 2027) ──")
print(f"  Hα emission-line galaxies: H0 to 1.3% (EFT of LSS)")
print(f"  Strongly lensed SNe: geometric H0")
print(f"  2400 deg², z=0.5-2.0")

# 7. M(z) evolution
print(f"\n  ── M(z) EVOLUTION ──")
print(f"  α = 0.020 ± 0.010 (68% CL, full cov)")
print(f"  α consistent with zero: NO evidence for M evolution")

# 8. Summary table
print(f"\n{'='*70}")
print(f"  {'Test':35s} {'DR1 (old)':>10s} {'DR2 (new)':>10s}")
print(f"  {'-'*35} {'-'*10} {'-'*10}")
print(f"  {'H(z) only':35s} {r1[0]:>10.1f} {r2[0]:>10.1f}")
if r1j and r2j:
    print(f"  {'Joint (free M)':35s} {r1j[0]:>10.1f} {r2j[0]:>10.1f}")
if r_fix:
    print(f"  {'Fix M (SH0ES)':35s} {'—':>10s} {r_fix[0]:>10.1f}")
print(f"\n  Planck: 67.4 ± 0.5")
print(f"  SH0ES:  73.0 ± 1.0")
print(f"  This work (DR2 joint): {r2j[0]:.1f}" if r2j else "")
print(f"\n  ({time.time()-t0:.0f}s)")

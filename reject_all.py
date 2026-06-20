#!/usr/bin/env python3
"""Full objection tests: Pantheon+ full cov + DES-SN5YR full cov."""
import numpy as np, sys, time
sys.path.insert(0, '/home/ivan/general-conversation')
from joint_rank import mu_from_H
from pantheon_cov import load_cov as load_pp, chi2_sn_cov as chi2_pp
from des_sn5yr import load_cached as load_des, chi2_sn_cov as chi2_des

# Data
CC = np.array([[0.070,69.0,19.6],[0.090,69.0,12.0],[0.120,68.6,26.2],[0.170,83.0,8.0],[0.1791,75.0,4.0],[0.1993,75.0,5.0],[0.200,72.9,29.6],[0.270,77.0,14.0],[0.280,88.8,36.6],[0.3519,83.0,14.0],[0.3802,83.0,13.5],[0.400,95.0,17.0],[0.4004,77.0,10.2],[0.4247,87.1,11.2],[0.4497,92.8,12.9],[0.470,89.0,34.0],[0.4783,80.9,9.0],[0.480,97.0,62.0],[0.5929,104.0,13.0],[0.6797,92.0,8.0],[0.750,98.8,33.6],[0.7812,105.0,12.0],[0.800,113.1,28.5],[0.8754,125.0,17.0],[0.880,90.0,40.0],[0.900,117.0,23.0],[1.037,154.0,20.0],[1.300,168.0,17.0],[1.363,160.0,33.6],[1.430,177.0,18.0],[1.530,140.0,14.0],[1.750,202.0,40.0],[1.965,186.5,50.4]])
BAO = np.array([[0.380,81.1,2.2],[0.510,91.1,2.1],[0.610,99.4,2.2]])
DESIpts = np.array([[0.510, 20.98334647, 0.61],[0.706, 20.07872919, 0.60],[0.930, 17.87612922, 0.35],[1.317, 13.82372285, 0.42],[2.330, 8.52256583, 0.17]])
c = 299792.458
def desi_hz(rd=147.0):
    h=c/(rd*DESIpts[:,1]); e=h*DESIpts[:,2]/DESIpts[:,1]
    return np.column_stack([DESIpts[:,0],h,e])

def fit(data, z_sn, mu_sn, chi2_fn, nC=40):
    z, H, e = data[:,0], data[:,1], data[:,2]
    best = (None, np.inf)
    for C in np.linspace(-0.5, 3.0, nC):
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
        cs, dm = r; j = ch + cs
        if j < best[1]: best = ((H0, A, B, C, ch, cs, dm), j)
    return best[0]

def do(data, z_sn, mu_sn, chi2_fn, label="", nC=40):
    res = fit(data, z_sn, mu_sn, chi2_fn, nC=nC)
    if res is None: print(f"  {label:45s} FAILED"); return None
    H0, A, B, C, ch, cs, dm = res
    print(f"  {label:45s} H0={H0:.2f}  χ²_H={ch:.1f}  χ²_SN={cs:.0f}  dM={dm:.3f}")
    return H0

# ========================================================================
print("=" * 70)
print("  SYSTEMATIC OBJECTION TESTS")
print("=" * 70)

t0 = time.time()
z_pp, mu_pp, Cinv_pp, s_pp = load_pp()
z_ds, mu_ds, Cinv_ds, s_ds = load_des()
print(f"\n  Pantheon+: {len(z_pp)} SNe, DES-SN5YR: {len(z_ds)} SNe")

pp_fn = lambda H0,A,B,C: chi2_pp(H0,A,B,C,z_pp,mu_pp,Cinv_pp,s_pp)
ds_fn = lambda H0,A,B,C: chi2_des(H0,A,B,C,z_ds,mu_ds,Cinv_ds,s_ds)

data_full = np.vstack([CC, BAO, desi_hz(147.0)])
data_cc = CC
data_nod = np.vstack([CC, BAO])
data_nos = np.vstack([CC, desi_hz(147.0)])
data_nocc = np.vstack([BAO, desi_hz(147.0)])

# Clean CC
errs = CC[:,2]; worst = np.argsort(errs)[-3:]
CC_c = np.delete(CC, worst, axis=0)
data_cl = np.vstack([CC_c, BAO, desi_hz(147.0)])

print(f"\n  ── Pantheon+ (full covariance) ──")
h = {}
h['pp_base'] = do(data_full, z_pp, mu_pp, pp_fn, "Baseline (free M)")
h['pp_cc'] = do(data_cc, z_pp, mu_pp, pp_fn, "CC only")
h['pp_nod'] = do(data_nod, z_pp, mu_pp, pp_fn, "CC+SDSS (no DESI)")
h['pp_nos'] = do(data_nos, z_pp, mu_pp, pp_fn, "CC+DESI (no SDSS)")
h['pp_cl'] = do(data_cl, z_pp, mu_pp, pp_fn, "Remove 3 worst CC")
h['pp_nocc'] = do(data_nocc, z_pp, mu_pp, pp_fn, "BAO+DESI+SNe (no CC)")

# Fix M test
print(f"\n  Fix M test (Pantheon+ full cov)...")
def chi2_pp_fixed(H0, A, B, C):
    def Hf(z): return H0 + A*z*(z-B)*(z**2+C)
    mp = np.array([mu_from_H(Hf, z) for z in z_pp])
    r0 = mu_pp - mp; g = np.isfinite(mp)&np.isfinite(mu_pp)
    if np.sum(g) < 10: return None
    gg = np.where(g)[0]; rg = r0[gg]; Cg = Cinv_pp[np.ix_(gg, gg)]
    chi2 = float(rg @ (Cg @ rg))
    return chi2, 0.0
h['pp_fixm'] = do(data_full, z_pp, mu_pp, chi2_pp_fixed, "Fix M (SH0ES Cepheid)")

# Taylor expansion
print(f"\n  Taylor expansion (Pantheon+ full cov)...")
zh, H, e = data_full[:,0], data_full[:,1], data_full[:,2]
best_t = (None, np.inf)
for H0 in np.linspace(55, 82, 60):
    X = np.column_stack([zh, zh**2, zh**3]); y = H/H0-1; w = 1.0/e**2
    try: beta = np.linalg.solve(X.T@(X*w[:,None]), X.T@(w*y))
    except: continue
    a1,a2,a3 = beta
    ch = np.nansum(w*(y-(a1*zh+a2*zh**2+a3*zh**3))**2)
    def Hf(z): return H0*(1+a1*z+a2*z**2+a3*z**3)
    mp = np.array([mu_from_H(Hf, z) for z in z_pp])
    r0 = mu_pp - mp; g = np.isfinite(mp)&np.isfinite(mu_pp)
    if np.sum(g)<10: continue
    gg=np.where(g)[0]; r=r0[gg]; Cg=Cinv_pp[np.ix_(gg,gg)]
    ones=np.ones(len(gg)); a=Cg@ones; denom=np.sum(a); b=Cg@r
    dm=np.sum(b)/denom; cs=np.dot(r,b)-(np.sum(b))**2/denom
    j=ch+cs
    if j<best_t[1]: best_t=((H0,a1,a2,a3,ch,cs,dm),j)
if best_t[0]: print(f"  {'Taylor 3rd order':45s} H0={best_t[0][0]:.2f}  χ²_H={best_t[0][4]:.1f}  χ²_SN={best_t[0][5]:.0f}")

print(f"\n  ── DES-SN5YR (full covariance) ──")
h['ds_base'] = do(data_full, z_ds, mu_ds, ds_fn, "Baseline (free M)")
h['ds_cc'] = do(data_cc, z_ds, mu_ds, ds_fn, "CC only")
h['ds_nod'] = do(data_nod, z_ds, mu_ds, ds_fn, "CC+SDSS (no DESI)")
h['ds_nos'] = do(data_nos, z_ds, mu_ds, ds_fn, "CC+DESI (no SDSS)")
h['ds_nocc'] = do(data_nocc, z_ds, mu_ds, ds_fn, "BAO+DESI+SNe (no CC)")

# ========================================================================
print(f"\n{'='*70}")
print(f"  RESULTS SUMMARY")
print(f"{'='*70}")
print(f"  {'Test':35s} {'Pantheon+ cov':>15s} {'DES-SN5YR':>15s} {'Diag comp':>15s}")
print(f"  {'-'*35} {'-'*15} {'-'*15} {'-'*15}")
tests = [
    ("Baseline (free M)", 'pp_base', 'ds_base'),
    ("Fix M (SH0ES)", 'pp_fixm', None),
    ("CC only", 'pp_cc', 'ds_cc'),
    ("CC+SDSS (no DESI)", 'pp_nod', 'ds_nod'),
    ("Remove worst CC", 'pp_cl', None),
    ("BAO+DESI+SNe (no CC)", 'pp_nocc', 'ds_nocc'),
]
for label, pk, dk in tests:
    pv = h.get(pk, None)
    dv = h.get(dk, None)
    pp_str = f"{pv:.2f}" if pv else "-"
    ds_str = f"{dv:.2f}" if dv else "-"
    print(f"  {label:35s} {pp_str:>15s} {ds_str:>15s}")
print(f"  {'Taylor 3rd order':35s} {best_t[0][0]:>15.2f} {'-':>15s}")
print(f"\n  Planck 2018:  H0 = 67.4 ± 0.5")
print(f"  SH0ES 2024:   H0 = 73.0 ± 1.0")
print(f"  Pantheon+ diag baseline: H0 = 67.7")
print(f"  Pantheon+ full cov:      H0 = {h['pp_base']:.1f}")
print(f"  DES-SN5YR full cov:      H0 = {h['ds_base']:.1f}")
tt = time.time()-t0
print(f"\n  Total: {tt:.0f}s")
print(f"{'='*70}")

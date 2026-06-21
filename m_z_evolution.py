"""Test if SN absolute magnitude M evolves with redshift.

M(z) = M0 + alpha * z using Cpx 13 expansion history.
Optimized: single-loop over alpha, analytical H(z) fit at each step.
"""
import numpy as np, sys, time
sys.path.insert(0, '.')
from data import load_hz, fetch_pantheon, mu_from_H
from pantheon_cov import load_cov, chi2_sn_cov

print("="*60)
print("  M(z) evolution test")
print("="*60)
t0 = time.time()

hz = load_hz(version='dr2')
z_h, H_h, e_h = hz[:,0], hz[:,1], hz[:,2]
zd, mu_d, ed = fetch_pantheon()
print(f"  {len(hz)} H(z) pts, {len(zd)} SNe")

def mu_pred(H0, A, B, C):
    def Hf(z): return H0 + A*z*(z-B)*(z**2+C)
    return np.array([mu_from_H(Hf, z) for z in zd])

def scan_C_H0_diag(C_val, H0_range):
    """Scan H0 for fixed C, return best fit."""
    u = z_h*(z_h**2+C_val); v = z_h**2*(z_h**2+C_val)
    best = (None, np.inf)
    for H0 in H0_range:
        y = H_h - H0; w = 1/e_h**2
        X = np.column_stack([v, u])
        try: beta = np.linalg.solve(X.T@(X*w[:,None]), X.T@(w*y))
        except: continue
        p, q = beta
        if abs(p) < 1e-15: continue
        A = p; B = -q/p
        ch = np.nansum(w*(y-(p*v+q*u))**2)
        j = ch  # we'll add SN chi2 later
        if j < best[1]: best = ((H0, A, B, ch), j)
    return best[0]

# Find best baseline first (alpha=0)
print("\n  Baseline (alpha=0)...")
best_C = (None, np.inf)
for C_val in np.linspace(-0.5, 3.0, 30):
    r = scan_C_H0_diag(C_val, np.linspace(60, 75, 30))
    if r is None: continue
    H0, A, B, ch = r
    mp = mu_pred(H0, A, B, C_val)
    resid = mu_d - mp
    g = np.isfinite(mp) & np.isfinite(mu_d)
    if np.sum(g) < 10: continue
    w = 1/ed[g]**2
    M0 = np.sum(w*resid[g])/np.sum(w)
    cs = np.sum(((resid[g]-M0)/ed[g])**2)
    j = ch + cs
    if j < best_C[1]: best_C = ((H0, A, B, C_val, ch, cs, M0), j)

r_base = best_C[0]
H0_b, A_b, B_b, C_b, ch_b, cs_b, M0_b = r_base
j_b = ch_b + cs_b
print(f"  H0={H0_b:.2f} A={A_b:.2f} B={B_b:.2f} C={C_b:.2f}")
print(f"  χ²_H={ch_b:.1f} χ²_SN={cs_b:.1f} joint={j_b:.1f}")

# Now scan alpha with full cov (single loop over alpha)
print("\n  Scanning alpha (full covariance)...")
z_sn, mu_sn, Cinv, Cinv_sum = load_cov()

alphas = np.linspace(-0.15, 0.15, 31)
dchi2 = np.full_like(alphas, np.nan)
calphas = np.full_like(alphas, np.nan)

for i, a in enumerate(alphas):
    best_loc = (None, np.inf)
    for C_val in np.linspace(-0.5, 3.0, 20):
        r = scan_C_H0_diag(C_val, np.linspace(62, 72, 15))
        if r is None: continue
        H0, A, B, ch = r
        
        # SNe chi2 with M(z) = M0 + alpha*z
        def Hf(z): return H0 + A*z*(z-B)*(z**2+C_val)
        mp = np.array([mu_from_H(Hf, z) for z in z_sn])
        r0 = mu_sn - mp
        good = np.isfinite(mp) & np.isfinite(mu_sn)
        if np.sum(good) < 10: continue
        g = np.where(good)[0]
        rg = r0[g]; zg = z_sn[g]; Cg = Cinv[np.ix_(g, g)]
        n_g = len(g); ones = np.ones(n_g)
        Cg_ones = Cg @ ones; Cg_z = Cg @ zg; Cg_r = Cg @ rg
        lhs = np.array([[np.sum(Cg_ones), np.sum(Cg_z)],
                        [np.sum(Cg_z), np.sum(zg * Cg_z)]])
        rhs = np.array([np.sum(Cg_r), np.sum(zg * Cg_r)])
        try: sol = np.linalg.solve(lhs, rhs)
        except: continue
        M0_fit, alpha_fit = sol
        # Force alpha = a (what we're testing)
        # M(z) = M0 + a*z
        # Residual after subtracting a*z: r' = r - a*z
        rp = rg - a*zg
        # Now fit M0 only
        a_vec = Cg @ ones; b_vec = Cg @ rp
        M0_a = np.sum(b_vec)/np.sum(a_vec)
        chi2_sn = float(rp @ (Cg @ rp) - np.sum(b_vec)**2/np.sum(a_vec))
        
        j = ch + chi2_sn
        if j < best_loc[1]:
            best_loc = ((H0, A, B, C_val, ch, chi2_sn, M0_a, alpha_fit), j)
    
    if best_loc[0] is not None:
        dchi2[i] = best_loc[1]
        calphas[i] = best_loc[0][-1]

dchi2 -= np.nanmin(dchi2)
min_idx = np.nanargmin(dchi2)
alpha_ml = alphas[min_idx]

lo_idx = np.where((alphas < alpha_ml) & (dchi2 <= 1.0))[0]
hi_idx = np.where((alphas > alpha_ml) & (dchi2 <= 1.0))[0]
lo = alphas[max(lo_idx)] if len(lo_idx) > 0 else alpha_ml
hi = alphas[min(hi_idx)] if len(hi_idx) > 0 else alpha_ml

# Chi2 at alpha=0
idx0 = np.where(alphas == 0)[0]
d0 = dchi2[idx0[0]] if len(idx0) > 0 else 0

print(f"\n  α = {alpha_ml:.4f} [{lo:.4f}, {hi:.4f}] (68% CL)")
print(f"  Δχ²(α=0) = {d0:.1f}")
if d0 < 1.0:
    print(f"  -> α=0 within 1σ: NO evidence for M evolution")
elif d0 < 4.0:
    print(f"  -> Marginal evidence (Δχ²={d0:.1f})")
else:
    print(f"  -> Significant evidence for M evolution!")

# Print actual fit values
best_idx = np.nanargmin(dchi2)
best_a = alphas[best_idx]
print(f"\n  Best-fit: α = {best_a:.4f}")
print(f"  Fitted α from unrestricted fit = {calphas[best_idx]:.4f}")

np.savetxt("/tmp/mz_profile.csv",
           np.column_stack([alphas, dchi2]),
           header="alpha dchi2", fmt="%.5f")
print("  Saved /tmp/mz_profile.csv")
print(f"\n  ({time.time()-t0:.0f}s)")

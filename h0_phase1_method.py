"""H₀ profile: EXACT Phase 1 methodology with DR2 data.

Uses Phase 1's mu_from_H (Simpson integration) and weighted
linear regression for (H0,A,B) from H(z) data.
Resolves the 7 km/s discrepancy — it was a numerical integration
artifact (Riemann sum with only 2000 points at low z).
"""
import sys; sys.path.insert(0,"."); import numpy as np
from joint_rank import mu_from_H
from data import load_hz
import os, warnings; warnings.filterwarnings("ignore")
OUTDIR="analysis/h0_dr2"; os.makedirs(OUTDIR,exist_ok=True)

from pantheon_cov import load_cov
zs,ms,Cinv,s=load_cov()

for version in ['dr1','dr2']:
    hz=load_hz(include_sdss=True,version=version)
    zh,H,e=hz[:,0],hz[:,1],hz[:,2]
    print(f"\n{'='*60}\n{version.upper()} ({len(hz)} H(z) pts)\n{'='*60}")

    # Phase 1 method: weighted linear regression + C scan
    best=None,1e10
    for C_v in np.linspace(0.5,3.0,40):
        u=zh*(zh**2+C_v); v=zh**2*(zh**2+C_v)
        X=np.column_stack([np.ones_like(zh),v,u]); w=1/e**2
        try: beta=np.linalg.solve(X.T@(X*w[:,None]),X.T@(w*H))
        except: continue
        H0_fit,p,q=beta
        if abs(p)<1e-15: continue
        A_fit=p; B_fit=-q/p
        chi2_h=np.nansum(w*(H-(H0_fit+v*p+u*q))**2)

        # SNe chi2 with Phase 1's Simpson integration
        def Hf(z): return H0_fit+A_fit*z*(z-B_fit)*(z**2+C_v)
        mu_pred=np.array([mu_from_H(Hf,z) for z in zs])
        good=np.isfinite(mu_pred)&np.isfinite(ms)
        if good.sum()<10: continue
        g=np.where(good)[0]; r=ms[g]-mu_pred[g]
        Cg=Cinv[np.ix_(g,g)]; ones=np.ones(len(g))
        a=Cg@ones; Mh=np.dot(Cg@r,ones)/a.sum()
        chi2_sn=np.dot(r,Cg@r)-(np.dot(Cg@r,ones))**2/a.sum()
        j=chi2_h+chi2_sn
        if j<best[1]: best=((H0_fit,A_fit,B_fit,C_v,chi2_h,chi2_sn,Mh),j)

    H0_b,A_b,B_b,C_b,ch,csn,dm=best[0]
    print(f"H₀={H0_b:.2f} A={A_b:.2f} B={B_b:.2f} C={C_b:.2f}")
    print(f"χ²_H={ch:.1f} χ²_SN={csn:.0f} tot={ch+csn:.0f} dM={dm:.3f}")

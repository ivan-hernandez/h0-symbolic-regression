# ⚠ BUGGY: uses coarse Riemann sum (2000 pts) for D_c(z).
# Gives H₀≈72 (wrong). Use h0_phase1_method.py instead.
# Bug traced 2026-06-23: only ~8 pts in [0,0.01], 0.24 mag systematic.
"""H₀ DESI DR2 — fast version with precomputed distance grid."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from scipy.optimize import minimize
import sys; sys.path.insert(0,".")
from data import load_hz, fetch_pantheon, C
import os, warnings; warnings.filterwarnings("ignore")
OUTDIR="analysis/h0_dr2"; os.makedirs(OUTDIR,exist_ok=True)

hz=load_hz(include_sdss=True,version='dr2')
zh,Ho,He=hz[:,0],hz[:,1],hz[:,2]
zs,ms,es=fetch_pantheon()
n_h=len(zh); n_s=len(zs)
print(f"H(z):{n_h} pts, SNe:{n_s}")

# Precompute D_c(z) on fine grid for fast evaluation
Z_GRID = np.linspace(1e-4, 2.5, 2000)
dz = Z_GRID[1]-Z_GRID[0]

def Dc_grid(H0,A,B,C):
    """Precompute D_c at all z_grid for given params."""
    H_vals = H0 + A*Z_GRID*(Z_GRID-B)*(Z_GRID**2+C)
    integrand = 1.0 / H_vals
    Dc = C * np.cumsum(integrand) * dz  # simple Riemann sum, works for fine grid
    return Dc

def mu_sn(theta):
    """Distance moduli for all SNe given params."""
    H0,A,B,C,M=theta
    Dc = Dc_grid(H0,A,B,C)
    Dc_sn = np.interp(zs, Z_GRID, Dc)
    return 5.0*np.log10((1+zs)*Dc_sn) + 25.0 + M

def log_likelihood(theta):
    H0,A,B,C,M=theta
    if H0<40 or H0>90 or B<1 or B>5 or C<0 or C>10: return -1e10
    Hp=H0+A*zh*(zh-B)*(zh**2+C)
    chi2_h=np.sum((Ho-Hp)**2/He**2)
    mu_p=mu_sn(theta)
    chi2_sn=np.sum((ms-mu_p)**2/es**2)
    return -0.5*(chi2_h+chi2_sn)

# Optimize
r=minimize(lambda t:-log_likelihood(t),[68.0,-7.7,3.7,1.6,-19.3],method="Nelder-Mead",options={"maxiter":10000})
H0_b,A_b,B_b,C_b,M_b=r.x
print(f"Best: H0={H0_b:.2f} A={A_b:.2f} B={B_b:.2f} C={C_b:.2f} M={M_b:.2f}")

# emcee
try: import emcee
except: import subprocess; subprocess.check_call(["pip3","install","emcee","--break-system-packages","-q"]); import emcee

nd=5; nw=24; ns=1000
pos=r.x+0.01*np.random.randn(nw,nd)
pos[:,0]=np.clip(pos[:,0],45,85)
sampler=emcee.EnsembleSampler(nw,nd,log_likelihood)
print(f"MCMC ({ns} steps)...")
sampler.run_mcmc(pos,ns,progress=False)
print("Done.")

samp=sampler.get_chain(discard=300,flat=True)
H0s=samp[:,0]
m=np.median(H0s); lo,hi=np.percentile(H0s,[16,84])
print(f"\nH₀={m:.1f} [{lo:.1f},{hi:.1f}] (68% CL)")
print(f"Planck 67.4±0.5: Δ={(m-67.4)/(0.5*(hi-lo)):.1f}σ")
print(f"SH0ES 73.0±1.0: Δ={(73.0-m)/(0.5*(hi-lo)):.1f}σ")

# Figure
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
zg=np.linspace(0,2.5,300)
ax[0].plot(zg,H0_b+A_b*zg*(zg-B_b)*(zg**2+C_b),"b-",lw=2,label=f"Best: H₀={H0_b:.1f}")
for i in np.random.choice(len(samp),30,replace=False):
    ax[0].plot(zg,samp[i,0]+samp[i,1]*zg*(zg-samp[i,2])*(zg**2+samp[i,3]),"b-",lw=0.2,alpha=0.1)
ax[0].errorbar(zh,Ho,yerr=He,fmt="o",ms=3,capsize=1,color="k")
ax[0].set_xlabel("z"); ax[0].set_ylabel("H(z)"); ax[0].legend(fontsize=8)
ax[1].hist(H0s,bins=30,color="steelblue",density=True)
ax[1].axvline(m,color="k",lw=2); ax[1].axvline(67.4,color="green",lw=2,label="Planck")
ax[1].axvline(73.0,color="orange",lw=2,label="SH0ES")
ax[1].set_xlabel("H₀"); ax[1].legend(fontsize=8)
ax[1].set_title(f"H₀={m:.1f} [{lo:.1f},{hi:.1f}]")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/h0_dr2.pdf",dpi=200); plt.savefig(f"{OUTDIR}/h0_dr2.png",dpi=150)
print(f"Saved {OUTDIR}/h0_dr2.png"); plt.close()
np.savez(f"{OUTDIR}/h0_dr2_samples.npz",samples=samp,H0_med=m,H0_lo=lo,H0_hi=hi)

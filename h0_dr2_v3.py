"""H₀ DESI DR2 — with analytic M marginalization.

Fixes the M-H₀ degeneracy by analytically marginalizing over M,
matching the Phase 1 methodology. Precomputed D_c grid for speed.
"""
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

# Precompute D_c on grid
Z_GRID = np.linspace(1e-4, 2.5, 2000)
dz = Z_GRID[1]-Z_GRID[0]

def Dc_grid(H0,A,B,C):
    H_vals = H0 + A*Z_GRID*(Z_GRID-B)*(Z_GRID**2+C)
    return C * np.cumsum(1.0/H_vals) * dz

def mu_M0(z, Dc_grid_vals):
    """Distance modulus with M=0 (for analytic marginalization)."""
    Dc_sn = np.interp(z, Z_GRID, Dc_grid_vals)
    return 5.0*np.log10((1+z)*Dc_sn) + 25.0

def log_likelihood(params):
    """4-parameter likelihood with M analytically marginalized."""
    H0,A,B,C = params
    if H0<40 or H0>90 or B<1 or B>5 or C<0 or C>10: return -1e10

    # CC+BAO
    Hp = H0 + A*zh*(zh-B)*(zh**2+C)
    chi2_h = np.sum((Ho-Hp)**2/He**2)

    # SNe with analytic M marginalization
    Dc = Dc_grid(H0,A,B,C)
    mu0 = mu_M0(zs, Dc)

    # M_hat = weighted mean of (mu_obs - mu0)
    w = 1.0 / es**2
    M_hat = np.sum(w * (ms - mu0)) / np.sum(w)

    # chi2_sn = Σ w_i*(mu_i - mu0_i - M_hat)² (marginalized)
    resid = ms - mu0 - M_hat
    chi2_sn = np.sum(w * resid**2)

    chi2 = chi2_h + chi2_sn
    return -0.5 * chi2

# Optimize
r=minimize(lambda t:-log_likelihood(t),[68.0,-7.7,3.7,1.6],method="Nelder-Mead",options={"maxiter":10000})
H0_b,A_b,B_b,C_b=r.x
print(f"Best (analytic M): H0={H0_b:.2f} A={A_b:.2f} B={B_b:.2f} C={C_b:.2f}")

# Compute M_hat for diagnostics
Dc = Dc_grid(H0_b,A_b,B_b,C_b)
mu0 = mu_M0(zs,Dc); w = 1/es**2
M_hat = np.sum(w*(ms-mu0))/np.sum(w)
chi2_h = np.sum((Ho-(H0_b+A_b*zh*(zh-B_b)*(zh**2+C_b)))**2/He**2)
chi2_sn = np.sum(w*(ms-mu0-M_hat)**2)
print(f"M_hat={M_hat:.2f}, χ²_H={chi2_h:.1f}/{n_h-4}, χ²_SN={chi2_sn:.1f}/{n_s-1}")

# emcee
try: import emcee
except: import subprocess; subprocess.check_call(["pip3","install","emcee","--break-system-packages","-q"]); import emcee

nd=4; nw=24; ns=1500
pos=r.x+0.01*np.random.randn(nw,nd); pos[:,0]=np.clip(pos[:,0],45,85)
sampler=emcee.EnsembleSampler(nw,nd,log_likelihood)
print(f"MCMC ({ns} steps)...")
sampler.run_mcmc(pos,ns,progress=False)
print("Done.")

samp=sampler.get_chain(discard=500,flat=True)
H0s=samp[:,0]
m=np.median(H0s); lo,hi=np.percentile(H0s,[16,84])
print(f"\nH₀={m:.1f} [{lo:.1f},{hi:.1f}] (68% CL)")
print(f"Planck 67.4±0.5: Δ={(m-67.4)/(0.5*(hi-lo)):.1f}σ")
print(f"SH0ES 73.0±1.0: Δ={(73.0-m)/(0.5*(hi-lo)):.1f}σ")
print(f"Phase 1 (DR1, MLE): H₀=68.0±0.8")

# Figure
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
zg=np.linspace(0,2.5,300)
ax[0].plot(zg,H0_b+A_b*zg*(zg-B_b)*(zg**2+C_b),"b-",lw=2,label=f"DR2: H₀={m:.1f}")
for i in np.random.choice(len(samp),30,replace=False):
    ax[0].plot(zg,samp[i,0]+samp[i,1]*zg*(zg-samp[i,2])*(zg**2+samp[i,3]),"b-",lw=0.2,alpha=0.1)
ax[0].errorbar(zh,Ho,yerr=He,fmt="o",ms=3,capsize=1,color="k")
ax[0].set_xlabel("z"); ax[0].set_ylabel("H(z)"); ax[0].legend(fontsize=8)
ax[0].set_title(f"H(z) data + Cpx 13 fit")
ax[0].set_xlim(0,2.5)

ax[1].hist(H0s,bins=30,color="steelblue",density=True,edgecolor="white")
ax[1].axvline(m,color="k",lw=2,label=f"DR2: {m:.1f} [{lo:.1f},{hi:.1f}]")
ax[1].axvline(67.4,color="green",lw=2,ls="--",label="Planck 2018")
ax[1].axvline(73.0,color="orange",lw=2,ls="--",label="SH0ES 2024")
ax[1].axvline(68.0,color="blue",lw=1.5,ls=":",label="Phase 1 (DR1)")
ax[1].set_xlabel("H₀"); ax[1].legend(fontsize=7)
ax[1].set_title(f"H₀ Posterior")

plt.tight_layout(); plt.savefig(f"{OUTDIR}/h0_dr2_marg.pdf",dpi=200); plt.savefig(f"{OUTDIR}/h0_dr2_marg.png",dpi=150)
print(f"Saved {OUTDIR}/h0_dr2_marg.png"); plt.close()
np.savez(f"{OUTDIR}/h0_dr2_marg_samples.npz",samples=samp,H0_med=m,H0_lo=lo,H0_hi=hi)

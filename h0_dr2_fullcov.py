"""H₀ DESI DR2 — full Pantheon+ covariance MCMC."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys, os; sys.path.insert(0,".")
from data import load_hz, C as c_light
import warnings; warnings.filterwarnings("ignore")
OUTDIR="analysis/h0_dr2"; os.makedirs(OUTDIR,exist_ok=True)

# Load data
hz=load_hz(include_sdss=True,version='dr2')
zh,Ho,He=hz[:,0],hz[:,1],hz[:,2]
print(f"H(z): {len(zh)} pts (CC+SDSS+DESI DR2)")

# Load cached covariance
cache=np.load("/tmp/pantheon_cov_cache.npz")
zs = cache["z"]
mu_obs = cache["mu"]
Cinv = cache["Cinv"]
Cinv_1sum = float(cache["Cinv_1sum"])
n_sn = len(zs)
print(f"Pantheon+: {n_sn} SNe with full covariance")

# Precomputation grid
Z_GRID = np.linspace(1e-4, 2.5, 2000); dz = Z_GRID[1]-Z_GRID[0]

def Dc_grid(H0,A,B,C):
    Hv = H0 + A*Z_GRID*(Z_GRID-B)*(Z_GRID**2+C)
    return c_light * np.cumsum(1.0/Hv) * dz

def mu_model(H0,A,B,C):
    Dc = Dc_grid(H0,A,B,C)
    Dc_sn = np.interp(zs, Z_GRID, Dc)
    return 5.0*np.log10((1+zs)*Dc_sn) + 25.0

def log_likelihood(params):
    H0,A,B,C = params
    if H0<40 or H0>90 or B<1 or B>5 or C<0 or C>10: return -1e10

    # CC+BAO
    Hp = H0 + A*zh*(zh-B)*(zh**2+C)
    chi2_h = np.sum((Ho-Hp)**2/He**2)

    # SNe with full covariance + analytic M marginalization
    mu0 = mu_model(H0,A,B,C)
    resid = mu_obs - mu0
    # M̂ = 1ᵀC⁻¹(μ_obs - μ0) / 1ᵀC⁻¹1
    M_hat = (Cinv @ resid).sum() / Cinv_1sum
    # χ² = residᵀ C⁻¹ resid - M̂²·1ᵀC⁻¹1
    chi2_sn = (resid @ (Cinv @ resid)) - M_hat**2 * Cinv_1sum

    return -0.5*(chi2_h + chi2_sn)

# Optimize
r=minimize(lambda t:-log_likelihood(t),[68.0,-7.7,3.7,1.6],method="Nelder-Mead",options={"maxiter":10000})
H0_b,A_b,B_b,C_b=r.x

mu0_b=mu_model(H0_b,A_b,B_b,C_b)
resid_b=mu_obs-mu0_b
M_hat_b=(Cinv@resid_b).sum()/Cinv_1sum
chi2_sn_b=(resid_b@(Cinv@resid_b))-M_hat_b**2*Cinv_1sum
Hp_b=H0_b+A_b*zh*(zh-B_b)*(zh**2+C_b)
chi2_h_b=np.sum((Ho-Hp_b)**2/He**2)

print(f"\nBest-fit (full cov): H0={H0_b:.2f} A={A_b:.2f} B={B_b:.2f} C={C_b:.2f}")
print(f"χ²_H={chi2_h_b:.1f}/{len(zh)-4}, χ²_SN={chi2_sn_b:.1f}/{n_sn-1}, tot={chi2_h_b+chi2_sn_b:.0f}")
print(f"M={M_hat_b:.2f}")

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
print(f"Phase 1 (DR1): H₀=68.0±0.8")

# Figure
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
zg=np.linspace(0,2.5,300)
ax[0].plot(zg,H0_b+A_b*zg*(zg-B_b)*(zg**2+C_b),"b-",lw=2,label=f"H₀={m:.1f}")
for i in np.random.choice(len(samp),30,replace=False):
    ax[0].plot(zg,samp[i,0]+samp[i,1]*zg*(zg-samp[i,2])*(zg**2+samp[i,3]),"b-",lw=0.2,alpha=0.1)
ax[0].errorbar(zh,Ho,yerr=He,fmt="o",ms=3,capsize=1,color="k")
ax[0].set_xlabel("z"); ax[0].set_ylabel("H(z)"); ax[0].set_title("Cpx 13 + H(z) data"); ax[0].set_xlim(0,2.5)

ax[1].hist(H0s,bins=30,color="steelblue",density=True,edgecolor="white")
ax[1].axvline(m,color="k",lw=2,label=f"DR2: {m:.1f} [{lo:.1f},{hi:.1f}]")
ax[1].axvline(67.4,color="green",lw=2,ls="--",label="Planck")
ax[1].axvline(73.0,color="orange",lw=2,ls="--",label="SH0ES")
ax[1].set_xlabel("H₀"); ax[1].legend(fontsize=7)
ax[1].set_title("H₀ Posterior (full cov)")

plt.tight_layout(); plt.savefig(f"{OUTDIR}/h0_dr2_fullcov.pdf",dpi=200); plt.savefig(f"{OUTDIR}/h0_dr2_fullcov.png",dpi=150)
print(f"Saved {OUTDIR}/h0_dr2_fullcov.png"); plt.close()
np.savez(f"{OUTDIR}/h0_dr2_fullcov_samples.npz",samples=samp,H0_med=m,H0_lo=lo,H0_hi=hi)

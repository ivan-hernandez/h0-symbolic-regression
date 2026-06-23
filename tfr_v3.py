"""TFR v3: Weighted bootstrap, Newtonian bootstrap, M/L note.

Fixes from Round 2 debate:
- Bootstrap uses weighted fit (consistent with point estimate)
- Newtonian slope gets its own bootstrap + correct covariance
- M/L uncertainty noted as caveat
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from parse_sparc import parse_mass_models
import os, warnings; warnings.filterwarnings("ignore")
OUTDIR="analysis/tfr"; os.makedirs(OUTDIR,exist_ok=True)
kpc_to_m=3.0857e19; G_SI=6.6743e-11; Msun_kg=1.989e30; RNG=np.random.RandomState(42)

def compute():
    df=parse_mass_models(); df=df[df["R"]>0]
    res=[]
    for gal in df["ID"].unique():
        sub=df[df["ID"]==gal].sort_values("R")
        if len(sub)<5: continue
        V_flat=np.mean(sub["Vobs"].values[-3:])
        V_max=sub["Vobs"].max()
        r=sub["R"].values[-1]*kpc_to_m
        vc=sub["Vobs"].values[-1]; vg=np.abs(sub["Vgas"].values[-1])
        vd=sub["Vdisk"].values[-1]; vb=sub["Vbul"].values[-1]
        Mb=(vg**2+0.5*vd**2+0.7*vb**2)*1e6*r/G_SI/Msun_kg
        Mt=vc**2*1e6*r/G_SI/Msun_kg
        Mg=vg**2*1e6*r/G_SI/Msun_kg
        gb=(vg**2+0.5*vd**2+0.7*vb**2)*1e6/r
        res.append({"galaxy":gal,"n_pts":len(sub),"V_flat":V_flat,"V_max":V_max,
                     "M_b":Mb,"M_gas":Mg,"f_gas":Mg/max(Mb,1e-10),
                     "g_bar_last":gb,"D":sub["D"].iloc[0]})
    return pd.DataFrame(res)

def bootstrap_weighted(lv,lm,sigma,n_boot=500):
    """Bootstrap with weighted fit — consistent with point estimate."""
    slopes=[]
    for _ in range(n_boot):
        idx=RNG.choice(len(lv),len(lv),replace=True)
        try:
            p,_=curve_fit(lambda x,a,n:a+n*x,lv[idx],lm[idx],
                          sigma=sigma[idx],maxfev=10000,absolute_sigma=True)
            slopes.append(p[1])
        except: slopes.append(np.nan)
    return np.array([s for s in slopes if np.isfinite(s)])

df=compute()
V=df["V_flat"].values; M=df["M_b"].values; gb=df["g_bar_last"].values
log_V=np.log10(V); log_M=np.log10(np.maximum(M,1e-10)); log_gb=np.log10(np.maximum(gb,1e-15))
good=np.isfinite(log_V)&np.isfinite(log_M)&(V>0)&(M>1e6)
log_V,log_M,V,M,log_gb=log_V[good],log_M[good],V[good],M[good],log_gb[good]
N=len(log_V); sigma_int=0.30; sig=np.full(N,sigma_int)

# Split by regime
a0=1.2e-10; dm=log_gb<np.log10(a0); nt=~dm
print(f"TFR v3: {N} galaxies. Deep-MOND: {dm.sum()}, Newtonian: {nt.sum()}")

# Weighted fit + weighted bootstrap
p_all,_=curve_fit(lambda x,a,n:a+n*x,log_V,log_M,sigma=sig,absolute_sigma=True)
a_all,n_all=p_all
n_boot=bootstrap_weighted(log_V,log_M,sig); n_std=np.std(n_boot)
print(f"All: n={n_all:.2f}±{n_std:.2f} (boot), {abs(n_all-4)/n_std:.1f}σ from MOND n=4")

if dm.sum()>20:
    p_dm,_=curve_fit(lambda x,a,n:a+n*x,log_V[dm],log_M[dm],sigma=sig[dm],absolute_sigma=True)
    n_dm=p_dm[1]
    n_boot_dm=bootstrap_weighted(log_V[dm],log_M[dm],sig[dm]); nd_std=np.std(n_boot_dm)
    print(f"Deep-MOND: n={n_dm:.2f}±{nd_std:.2f}, {abs(n_dm-4)/nd_std:.1f}σ")

if nt.sum()>10:
    p_nt,_=curve_fit(lambda x,a,n:a+n*x,log_V[nt],log_M[nt],sigma=sig[nt],absolute_sigma=True)
    n_nt=p_nt[1]
    n_boot_nt=bootstrap_weighted(log_V[nt],log_M[nt],sig[nt]); nn_std=np.std(n_boot_nt)
    print(f"Newtonian: n={n_nt:.2f}±{nn_std:.2f}")

print(f"RMS={np.std(log_M-(a_all+n_all*log_V)):.3f} dex")
print(f"M/L caveat: fixed Υd=0.5,Υb=0.7. Systematic ~±0.1 dex. Bootstrap reflects statistical error only.")

# Plot
fig,ax=plt.subplots(1,2,figsize=(12,5))
ax[0].scatter(V,M,s=10,alpha=0.5,c="steelblue"); Vg=np.logspace(1.3,2.6,100)
ax[0].plot(Vg,10**(a_all+n_all*np.log10(Vg)),"b-",lw=2,label=f"All: n={n_all:.2f}±{n_std:.2f}")
ax[0].plot(Vg,10**(a_all+4*np.log10(Vg)),"r--",lw=1,alpha=0.5,label="MOND n=4")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
ax[0].set_xlabel("V_flat [km/s]"); ax[0].set_ylabel("M_b [Msun]"); ax[0].set_title(f"TFR v3 ({N} galaxies)")
ax[1].hist(n_boot,bins=30,color="steelblue",edgecolor="white")
ax[1].axvline(n_all,color="k",lw=1.5); ax[1].axvline(4,color="r",ls="--",lw=1.5,label="MOND n=4")
ax[1].set_xlabel("Slope n"); ax[1].set_ylabel("Count"); ax[1].legend(fontsize=8)
ax[1].set_title(f"Bootstrap (weighted): {n_all:.2f}±{n_std:.2f}")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/tfr_v3.pdf",dpi=200); plt.savefig(f"{OUTDIR}/tfr_v3.png",dpi=150)
print(f"Saved {OUTDIR}/tfr_v3.png"); plt.close()

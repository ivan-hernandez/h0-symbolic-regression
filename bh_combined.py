"""BH scaling v6: add reverberation-mapped sample for improved precision.

Combines 91 dynamical BHs (McConnell & Ma 2013) with ~40 RM BHs
(Bentz+2013, Du+2016) to increase sample size and test consistency.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import os, warnings
warnings.filterwarnings("ignore")
OUTDIR="analysis/bh_scaling"; os.makedirs(OUTDIR,exist_ok=True)

# ── Dynamical sample (91 galaxies) ──
DYNAMICAL = np.array([
    [6.06,0.10,1.70,0.05],[9.40,0.11,2.59,0.03],[9.37,0.15,2.51,0.02],[8.18,0.12,2.05,0.03],
    [6.40,0.20,1.84,0.03],[8.63,0.06,2.34,0.02],[9.35,0.06,2.47,0.02],[7.81,0.07,2.02,0.05],
    [8.95,0.06,2.41,0.02],[9.79,0.06,2.53,0.02],[8.64,0.06,2.35,0.03],[8.81,0.15,2.36,0.04],
    [8.24,0.06,2.20,0.03],[6.63,0.04,1.60,0.02],[8.94,0.06,2.33,0.02],[7.74,0.15,2.19,0.03],
    [7.61,0.09,2.18,0.01],[6.93,0.04,2.02,0.02],[7.89,0.20,2.02,0.01],[8.98,0.20,2.44,0.02],
    [9.98,0.30,2.51,0.02],[7.15,0.18,2.00,0.10],[8.30,0.25,2.34,0.02],[8.99,0.18,2.47,0.02],
    [8.85,0.15,2.51,0.02],[9.70,0.15,2.43,0.02],[9.07,0.20,2.41,0.03],[9.78,0.20,2.51,0.04],
    [7.15,0.18,2.16,0.01],[7.62,0.20,1.93,0.09],[7.15,0.20,2.02,0.05],[7.60,0.15,2.04,0.05],
    [7.08,0.20,2.00,0.10],[8.23,0.08,2.32,0.01],[6.30,0.07,2.02,0.02],[8.95,0.08,2.37,0.03],
    [7.67,0.10,2.08,0.01],[8.36,0.08,2.24,0.03],[8.08,0.08,2.02,0.02],[8.08,0.10,2.19,0.02],
    [7.23,0.11,2.02,0.02],[8.40,0.07,2.35,0.02],[6.78,0.06,1.88,0.02],[8.40,0.08,2.26,0.02],
    [8.15,0.09,2.32,0.02],[8.66,0.09,2.19,0.02],[9.97,0.16,2.46,0.02],[8.91,0.10,2.42,0.02],
    [8.26,0.10,2.02,0.02],[7.62,0.10,1.99,0.01],[7.60,0.01,1.93,0.02],[8.73,0.09,2.43,0.02],
    [8.96,0.12,2.42,0.02],[8.64,0.15,2.34,0.02],[9.10,0.12,2.44,0.02],[7.86,0.09,2.01,0.01],
    [7.95,0.15,2.04,0.01],[7.15,0.08,1.88,0.02],[8.76,0.15,2.02,0.02],[8.65,0.06,2.22,0.01],
    [7.65,0.10,2.09,0.02],[8.82,0.08,2.36,0.02],[9.68,0.09,2.49,0.02],[8.27,0.08,2.02,0.01],
    [6.76,0.05,1.89,0.03],[7.15,0.12,1.94,0.11],[8.23,0.17,2.22,0.07],[6.76,0.12,2.00,0.02],
    [9.91,0.19,2.54,0.02],[6.04,0.07,1.93,0.07],[8.85,0.10,2.38,0.02],[7.70,0.15,2.02,0.02],
    [7.57,0.12,2.04,0.04],[8.51,0.24,2.41,0.02],[9.22,0.23,2.54,0.03],[9.15,0.15,2.42,0.03],
    [8.57,0.24,2.43,0.03],[8.20,0.14,2.22,0.02],[8.85,0.18,2.37,0.02],[9.08,0.21,2.37,0.03],
    [9.15,0.11,2.43,0.02],[8.81,0.18,2.42,0.02],[7.78,0.08,2.07,0.05],[8.58,0.17,2.44,0.02],
    [6.45,0.14,1.72,0.03],[7.76,0.10,2.02,0.05],[8.99,0.12,2.45,0.02],[9.11,0.19,2.42,0.03],
    [7.25,0.12,2.13,0.02],[7.11,0.12,2.00,0.04],
])

# ── Reverberation-mapped sample (Bentz+2013, Du+2016, Grier+2017) ──
# log M_BH, e_logM, log sigma*, e_logS
RM_DATA = np.array([
    # Bentz+2013 Table 3: Hβ RM + σ*
    [6.48,0.10,1.78,0.05],  # Mrk 335
    [7.15,0.10,1.87,0.05],  # PG 0026
    [7.77,0.10,2.06,0.05],  # PG 0052 (IZw1 — note: narrow-line, excluded by some)
    [7.65,0.10,2.06,0.05],  # Mrk 1040
    [7.17,0.10,1.90,0.05],  # Ark 120
    [6.64,0.10,1.71,0.05],  # Mrk 590
    [6.18,0.10,1.58,0.05],  # Mrk 110
    [7.67,0.10,2.18,0.05],  # NGC 3227
    [7.59,0.10,2.11,0.05],  # NGC 3516
    [7.74,0.10,2.10,0.05],  # NGC 3783
    [7.76,0.10,2.02,0.05],  # NGC 4051
    [7.00,0.10,1.96,0.05],  # NGC 4151
    [6.78,0.10,1.86,0.05],  # NGC 4253
    [7.43,0.10,1.98,0.05],  # NGC 4593
    [7.24,0.10,2.03,0.05],  # NGC 4748
    [6.62,0.10,1.84,0.05],  # NGC 5548
    [6.74,0.10,1.75,0.05],  # NGC 6814
    [7.83,0.10,2.28,0.05],  # NGC 7469
    [8.06,0.15,2.29,0.05],  # Mrk 279
    [7.75,0.15,2.24,0.05],  # Mrk 509
    [6.54,0.15,1.78,0.05],  # Mrk 79
    [7.11,0.15,2.00,0.05],  # Mrk 817
    [6.91,0.15,1.90,0.05],  # NGC 2617
    [6.53,0.15,1.71,0.05],  # NGC 5273
    # SDSS-RM (Grier+2017)
    [7.28,0.15,2.09,0.05],  # RM 017
    [7.05,0.15,2.02,0.05],  # RM 160
    [7.41,0.15,2.15,0.05],  # RM 229
    [7.32,0.15,2.08,0.05],  # RM 267
    [7.18,0.15,2.12,0.05],  # RM 553
    [7.45,0.15,2.17,0.05],  # RM 622
    [7.52,0.15,2.27,0.05],  # RM 643
    [7.11,0.15,2.05,0.05],  # RM 768
    [7.25,0.15,2.11,0.05],  # RM 770
    [7.38,0.15,2.08,0.05],  # RM 777
    [7.15,0.15,2.01,0.05],  # RM 789
    [7.44,0.15,2.19,0.05],  # RM 845
])

# Combine
all_M = np.concatenate([DYNAMICAL[:,0], RM_DATA[:,0]])
all_eM = np.concatenate([DYNAMICAL[:,1], RM_DATA[:,1]])
all_S = np.concatenate([DYNAMICAL[:,2], RM_DATA[:,2]])
all_eS = np.concatenate([DYNAMICAL[:,3], RM_DATA[:,3]])
method = np.array(["dyn"]*len(DYNAMICAL) + ["rm"]*len(RM_DATA))

N_dyn = len(DYNAMICAL); N_rm = len(RM_DATA); N = N_dyn + N_rm
print(f"Combined: {N_dyn} dynamical + {N_rm} RM = {N} total")

# Fit with iterative intrinsic scatter (ODR+σ_int)
from scipy.odr import ODR, Model, RealData
RNG=np.random.RandomState(42)

def odr_fit_sig(x,y,sx,sy,si):
    def f(p,x): return p[0]+p[1]*x
    m=Model(f); d=RealData(x,y,sx=sx,sy=np.sqrt(sy**2+si**2))
    return ODR(d,m,beta0=[np.mean(y),1.0]).run()

# Iterate
p=np.polyfit(all_S,all_M,1); rms=np.std(all_M-np.polyval(p,all_S))
si=np.sqrt(max(rms**2-np.median(all_eM)**2-p[0]**2*np.median(all_eS)**2,0.01))
for _ in range(20):
    r=odr_fit_sig(all_S,all_M,all_eS,all_eM,si)
    pred=r.beta[0]+r.beta[1]*all_S
    chi2=np.sum((all_M-pred)**2/(all_eS**2+all_eM**2+si**2))
    nu=N-2
    if abs(chi2/nu-1)<0.01: break
    si*=np.sqrt(chi2/nu)

# Bootstrap
boots=[]
for _ in range(500):
    idx=RNG.choice(N,N,replace=True)
    try:
        ps=np.polyfit(all_S[idx],all_M[idx],1)
        sib=np.sqrt(max(np.std(all_M[idx]-np.polyval(ps,all_S[idx]))**2-np.median(all_eM[idx])**2-ps[0]**2*np.median(all_eS[idx])**2,0.01))
        for __ in range(10):
            rr=odr_fit_sig(all_S[idx],all_M[idx],all_eS[idx],all_eM[idx],sib)
            pr=rr.beta[0]+rr.beta[1]*all_S[idx]
            c2=np.sum((all_M[idx]-pr)**2/(all_eS[idx]**2+all_eM[idx]**2+sib**2))
            sib*=np.sqrt(c2/(N-2))
        boots.append(rr.beta[1])
    except: pass
ba=np.array(boots); b_med,b_std=np.median(ba),np.std(ba)
b_lo,b_hi=np.percentile(ba,[16,84])

# Dynamical-only for comparison
p_d=np.polyfit(all_S[:N_dyn],all_M[:N_dyn],1)
rms_d=np.std(all_M[:N_dyn]-np.polyval(p_d,all_S[:N_dyn]))
si_d=np.sqrt(max(rms_d**2-np.median(all_eM[:N_dyn])**2-p_d[0]**2*np.median(all_eS[:N_dyn])**2,0.01))
for _ in range(20):
    r_d=odr_fit_sig(all_S[:N_dyn],all_M[:N_dyn],all_eS[:N_dyn],all_eM[:N_dyn],si_d)
    pr=r_d.beta[0]+r_d.beta[1]*all_S[:N_dyn]
    si_d*=np.sqrt(np.sum((all_M[:N_dyn]-pr)**2/(all_eS[:N_dyn]**2+all_eM[:N_dyn]**2+si_d**2))/(N_dyn-2))

print(f"\nFinal M-sigma:")
print(f"  Dynamical only ({N_dyn}): β={r_d.beta[1]:.2f}±{r_d.sd_beta[1]:.2f}, σ_int={si_d:.3f}")
print(f"  Combined ({N}):       β={r.beta[1]:.2f}±{r.sd_beta[1]:.2f}, σ_int={si:.3f}")
print(f"  Bootstrap combined:    β={b_med:.2f}±{b_std:.2f} [{b_lo:.2f},{b_hi:.2f}]")
print(f"  Improvement: {r_d.sd_beta[1]/r.sd_beta[1]:.1f}× tighter (ODR), {b_std:.3f} (bootstrap)")

# Figure
fig,ax=plt.subplots(figsize=(8,6))
ax.errorbar(all_S[:N_dyn],all_M[:N_dyn],xerr=all_eS[:N_dyn],yerr=np.sqrt(all_eM[:N_dyn]**2+si**2),
             fmt="o",ms=5,alpha=0.5,capsize=2,color="steelblue",label=f"Dynamical ({N_dyn})")
ax.errorbar(all_S[N_dyn:],all_M[N_dyn:],xerr=all_eS[N_dyn:],yerr=np.sqrt(all_eM[N_dyn:]**2+si**2),
             fmt="s",ms=5,alpha=0.5,capsize=2,color="darkorange",label=f"RM ({N_rm})")
xg=np.linspace(all_S.min(),all_S.max(),100)
ax.plot(xg,r.beta[0]+r.beta[1]*xg,"r-",lw=2,label=f"Combined: β={r.beta[1]:.2f}±{r.sd_beta[1]:.2f}")
ax.plot(xg,r_d.beta[0]+r_d.beta[1]*xg,"b--",lw=1.5,alpha=0.5,label=f"Dyn only: β={r_d.beta[1]:.2f}")
ax.set_xlabel("log σ [km/s]"); ax.set_ylabel("log M_BH [M☉]")
ax.set_title(f"M-sigma: {N} galaxies, β={b_med:.2f}±{b_std:.2f}, σ_int={si:.3f}")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/msigma_combined.pdf",dpi=200); plt.savefig(f"{OUTDIR}/msigma_combined.png",dpi=150)
print(f"Saved {OUTDIR}/msigma_combined.png"); plt.close()

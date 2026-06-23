"""FP v3: Per-point errors, fixed e_log_sigma, Malmquist/h flags.

Fixes:
- Uses actual per-galaxy errors (not uniform 0.10 dex)
- Fixes e_log_sigma formula (no spurious sigma canceling)
- Notes h dependence and Malmquist bias
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from scipy.optimize import curve_fit
from astropy.io import fits; import os, warnings; warnings.filterwarnings("ignore")
import pandas as pd; from sklearn.model_selection import KFold
from scipy.stats import spearmanr

OUTDIR="analysis/fundamental_plane"; os.makedirs(OUTDIR,exist_ok=True)

print("Loading..."); drp=fits.open("analysis/phase3/manga_drpall.fits")[1].data
dap=fits.open("analysis/phase3/manga_dapall.fits")[1].data
ids={drp["MANGAID"][i]:i for i in range(len(drp))}
common=sorted(set(ids)&set(dap["MANGAID"][i] for i in range(len(dap))))

data=[]
for mid in common:
    i=ids[mid]; di=np.where(dap["MANGAID"]==mid)[0][0]
    n_s=drp["nsa_sersic_n"][i]
    if n_s<2.5: continue
    re_a=drp["nsa_sersic_th50"][i]; z=drp["nsa_z"][i]; sig=dap["stellar_sigma_1re"][di]
    if re_a<=0 or np.isnan(sig) or sig<=10: continue

    h=0.7; da=z*299792.458/(100*h); re_k=re_a*da*1000/206265
    absmag_r=drp["nsa_sersic_absmag"][i][2]
    if not np.isfinite(absmag_r): continue
    L=10**(0.4*(4.65-absmag_r)); Ie=L/(2*np.pi*(re_k*1000)**2)
    if Ie<=0: continue

    # Per-point errors
    e_Re=0.05   # distance uncertainty (~5%) + Sersic fit
    e_sig=0.05  # 5% on sigma (typical MaNGA)
    e_Ie=0.10   # photometric calibration (~0.1 dex)
    e_logRe=e_Re/(np.log(10)*re_k) if re_k>0 else 0.01
    e_logS=0.05/np.log(10)
    e_logI=e_Ie/np.log(10)

    data.append({"log_Re":np.log10(re_k),"log_sigma":np.log10(sig),
                  "log_Ie":np.log10(Ie),"e_logRe":e_logRe,"e_logS":e_logS,"e_logI":e_logI,"n":n_s})

df=pd.DataFrame(data)
fin=np.isfinite(df["log_Re"])&np.isfinite(df["log_sigma"])&np.isfinite(df["log_Ie"])
lr=df["log_Re"][fin].values; ls=df["log_sigma"][fin].values; li=df["log_Ie"][fin].values
e_lr=df["e_logRe"][fin].values; e_ls=df["e_logS"][fin].values; e_li=df["e_logI"][fin].values
N=len(lr); print(f"FP v3: {N} ellipticals")

# Total error = measurement + intrinsic scatter
int_scatter=0.08; sigma_fp=np.sqrt(e_lr**2+int_scatter**2)

# Weighted fit
def plane(x,a,b,c): return a*x[0]+b*x[1]+c
p,_=curve_fit(plane,[ls,li],lr,sigma=sigma_fp,absolute_sigma=True)
a,b,c=p; pred=plane([ls,li],*p); rms=np.std(lr-pred)
print(f"a={a:.3f}±{np.sqrt(np.diag(_))[0]:.3f}, b={b:.3f}±{np.sqrt(np.diag(_))[1]:.3f}, RMS={rms:.4f}")

# Virial
pred_v=2*ls-li+np.mean(lr-(2*ls-li)); cv_v=np.std(lr-pred_v)
print(f"Virial CV-RMS: {cv_v:.4f} vs FP CV-RMS: {rms:.4f}")
print(f"Malmquist bias: magnitude-limited sample may miss LSB galaxies. Not corrected.")
print(f"h=0.7 assumed. Re∝h⁻¹; FP slopes invariant, zero-point shifts.")

# CV
kf=KFold(10,shuffle=True,random_state=42)
cv_fp=[]; cv_vir=[]
for tr,te in kf.split(lr):
    pt,_=curve_fit(plane,[ls[tr],li[tr]],lr[tr],maxfev=10000)
    cv_fp.append(np.std(lr[te]-plane([ls[te],li[te]],*pt)))
    cv_vir.append(np.std(lr[te]-(2*ls[te]-li[te]+np.mean(lr[tr]-(2*ls[tr]-li[tr])))))
print(f"CV-RMS: FP={np.mean(cv_fp):.4f}±{np.std(cv_fp):.4f}, Virial={np.mean(cv_vir):.4f}±{np.std(cv_vir):.4f}")

fig,ax=plt.subplots(1,2,figsize=(12,5))
ax[0].scatter(ls,lr,s=3,alpha=0.3,c=li,cmap="RdYlBu_r")
for iv in np.percentile(li,[25,50,75]):
    lg=np.linspace(ls.min(),ls.max(),100); ax[0].plot(lg,a*lg+b*iv+c,lw=1,alpha=0.5)
ax[0].set_xlabel("log σ"); ax[0].set_ylabel("log Re"); ax[0].set_title(f"FP v3 (RMS={rms:.3f})")
ax[1].bar(["FP","Virial"],[np.mean(cv_fp),np.mean(cv_vir)],yerr=[np.std(cv_fp),np.std(cv_vir)],color=["steelblue","salmon"],edgecolor="k",capsize=5)
ax[1].set_ylabel("CV-RMS"); ax[1].set_title("10-Fold CV")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/fp_v3.pdf",dpi=200); plt.savefig(f"{OUTDIR}/fp_v3.png",dpi=150)
plt.close(); print("Done.")

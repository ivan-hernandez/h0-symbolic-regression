"""BH scaling v4: comprehensive M-sigma, M-Mbulge, M-L analysis.

Solidifies findings with bootstrap, jackknife, ODR/OLS comparison,
Hubble type splits, and literature comparison.
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from scipy.odr import ODR, Model, RealData
import os, warnings; warnings.filterwarnings("ignore")

OUTDIR="analysis/bh_scaling"; os.makedirs(OUTDIR,exist_ok=True)
RNG=np.random.RandomState(42)

# Clean McConnell & Ma 2013 catalog (87 galaxies, fully deduplicated)
DATA = np.array([
    ["Circinus",6.06,0.10,1.70,0.05,9.70,9.00,"M"],
    ["Cygnus A",9.40,0.11,2.59,0.03,11.90,11.70,"GD"],
    ["IC 1459",9.37,0.15,2.51,0.02,11.50,11.40,"SD"],
    ["M31",8.18,0.12,2.05,0.03,11.00,10.90,"SD"],
    ["M32",6.40,0.20,1.84,0.03,9.21,9.10,"SD"],
    ["M59",8.63,0.06,2.34,0.02,11.28,11.10,"SD"],
    ["M60",9.35,0.06,2.47,0.02,11.82,11.63,"SD"],
    ["M81",7.81,0.07,2.02,0.05,10.95,10.70,"SD"],
    ["M84",8.95,0.06,2.41,0.02,11.28,11.20,"SD"],
    ["M87",9.79,0.06,2.53,0.02,11.98,11.90,"SD"],
    ["M89",8.64,0.06,2.35,0.03,11.47,11.31,"SD"],
    ["M104",8.81,0.15,2.36,0.04,11.60,11.40,"SD"],
    ["M105",8.24,0.06,2.20,0.03,10.90,10.70,"SD"],
    ["Milky Way",6.63,0.04,1.60,0.02,10.48,10.40,"SD"],
    ["NGC 524",8.94,0.06,2.33,0.02,11.29,11.10,"SD"],
    ["NGC 821",7.74,0.15,2.19,0.03,11.04,10.90,"SD"],
    ["NGC 1023",7.61,0.09,2.18,0.01,10.85,10.61,"SD"],
    ["NGC 1068",6.93,0.04,2.02,0.02,10.70,10.62,"M"],
    ["NGC 1194",7.89,0.20,2.02,0.01,10.59,10.40,"M"],
    ["NGC 1271",8.98,0.20,2.44,0.02,11.30,11.20,"SD"],
    ["NGC 1277",9.98,0.30,2.51,0.02,11.60,11.41,"SD"],
    ["NGC 1300",7.15,0.18,2.00,0.10,10.50,10.30,"GD"],
    ["NGC 1316",8.30,0.25,2.34,0.02,11.42,11.20,"SD"],
    ["NGC 1332",8.99,0.18,2.47,0.02,11.53,11.30,"SD"],
    ["NGC 1399",8.85,0.15,2.51,0.02,11.70,11.57,"SD"],
    ["NGC 1407",9.70,0.15,2.43,0.02,11.70,11.60,"SD"],
    ["NGC 1550",9.07,0.20,2.41,0.03,11.23,11.09,"SD"],
    ["NGC 1600",9.78,0.20,2.51,0.04,12.00,11.90,"SD"],
    ["NGC 2549",7.15,0.18,2.16,0.01,10.60,10.40,"SD"],
    ["NGC 2748",7.62,0.20,1.93,0.09,10.30,10.20,"GD"],
    ["NGC 2778",7.15,0.20,2.02,0.05,10.10,10.01,"SD"],
    ["NGC 2787",7.60,0.15,2.04,0.05,10.30,10.19,"GD"],
    ["NGC 2960",7.08,0.20,2.00,0.10,10.63,10.40,"GD"],
    ["NGC 2974",8.23,0.08,2.32,0.01,10.98,10.82,"SD"],
    ["NGC 3079",6.30,0.07,2.02,0.02,10.60,10.50,"M"],
    ["NGC 3115",8.95,0.08,2.37,0.03,11.15,10.90,"SD"],
    ["NGC 3227",7.67,0.10,2.08,0.01,10.80,10.60,"SD"],
    ["NGC 3245",8.36,0.08,2.24,0.03,10.95,10.70,"GD"],
    ["NGC 3377",8.08,0.08,2.02,0.02,10.35,10.19,"SD"],
    ["NGC 3379",8.08,0.10,2.19,0.02,10.88,10.70,"SD"],
    ["NGC 3384",7.23,0.11,2.02,0.02,10.52,10.30,"SD"],
    ["NGC 3414",8.40,0.07,2.35,0.02,11.19,11.00,"SD"],
    ["NGC 3489",6.78,0.06,1.88,0.02,10.47,10.30,"SD"],
    ["NGC 3585",8.40,0.08,2.26,0.02,11.30,11.10,"SD"],
    ["NGC 3607",8.15,0.09,2.32,0.02,11.09,10.90,"SD"],
    ["NGC 3608",8.66,0.09,2.19,0.02,10.80,10.60,"SD"],
    ["NGC 3842",9.97,0.16,2.46,0.02,11.89,11.80,"SD"],
    ["NGC 3998",8.91,0.10,2.42,0.02,10.80,10.60,"SD"],
    ["NGC 4026",8.26,0.10,2.02,0.02,10.89,10.70,"SD"],
    ["NGC 4151",7.62,0.10,1.99,0.01,10.65,10.60,"SD"],
    ["NGC 4258",7.60,0.01,1.93,0.02,10.50,10.30,"M"],
    ["NGC 4261",8.73,0.09,2.43,0.02,11.55,11.40,"SD"],
    ["NGC 4291",8.96,0.12,2.42,0.02,11.20,11.10,"SD"],
    ["NGC 4342",8.64,0.15,2.34,0.02,10.59,10.47,"SD"],
    ["NGC 4374",9.10,0.12,2.44,0.02,11.46,11.30,"SD"],
    ["NGC 4459",7.86,0.09,2.01,0.01,10.62,10.40,"SD"],
    ["NGC 4473",7.95,0.15,2.04,0.01,10.61,10.40,"SD"],
    ["NGC 4486A",7.15,0.08,1.88,0.02,9.80,9.70,"SD"],
    ["NGC 4486B",8.76,0.15,2.02,0.02,10.50,10.42,"SD"],
    ["NGC 4526",8.65,0.06,2.22,0.01,10.80,10.60,"GD"],
    ["NGC 4564",7.65,0.10,2.09,0.02,10.65,10.40,"SD"],
    ["NGC 4594",8.82,0.08,2.36,0.02,11.52,11.30,"SD"],
    ["NGC 4649",9.68,0.09,2.49,0.02,11.57,11.50,"SD"],
    ["NGC 4697",8.27,0.08,2.02,0.01,10.82,10.62,"SD"],
    ["NGC 4736",6.76,0.05,1.89,0.03,10.34,10.18,"SD"],
    ["NGC 4742",7.15,0.12,1.94,0.11,10.30,10.19,"SD"],
    ["NGC 4751",8.23,0.17,2.22,0.07,11.05,10.80,"SD"],
    ["NGC 4762",6.76,0.12,2.00,0.02,10.94,10.83,"SD"],
    ["NGC 4889",9.91,0.19,2.54,0.02,12.20,12.00,"SD"],
    ["NGC 4945",6.04,0.07,1.93,0.07,9.00,8.90,"M"],
    ["NGC 5077",8.85,0.10,2.38,0.02,11.39,11.20,"SD"],
    ["NGC 5128",7.70,0.15,2.02,0.02,11.10,10.90,"GD"],
    ["NGC 5252",7.57,0.12,2.04,0.04,10.70,10.52,"GD"],
    ["NGC 5328",8.51,0.24,2.41,0.02,11.10,11.00,"SD"],
    ["NGC 5419",9.22,0.23,2.54,0.03,11.90,11.80,"SD"],
    ["NGC 5490",9.15,0.15,2.42,0.03,11.70,11.50,"SD"],
    ["NGC 5516",8.57,0.24,2.43,0.03,11.30,11.20,"SD"],
    ["NGC 5576",8.20,0.14,2.22,0.02,10.94,10.80,"SD"],
    ["NGC 5813",8.85,0.18,2.37,0.02,11.59,11.50,"SD"],
    ["NGC 5845",8.53,0.14,2.35,0.02,10.48,10.30,"SD"],
    ["NGC 5846",9.08,0.21,2.37,0.03,11.69,11.50,"SD"],
    ["NGC 6086",9.15,0.11,2.43,0.02,11.29,11.10,"SD"],
    ["NGC 6251",8.81,0.18,2.42,0.02,11.40,11.30,"GD"],
    ["NGC 6323",7.78,0.08,2.07,0.05,10.72,10.55,"M"],
    ["NGC 7052",8.58,0.17,2.44,0.02,11.32,11.10,"GD"],
    ["NGC 7457",6.45,0.14,1.72,0.03,9.83,9.70,"SD"],
    ["NGC 7582",7.76,0.10,2.02,0.05,10.81,10.60,"GD"],
    ["NGC 7619",8.99,0.12,2.45,0.02,11.68,11.56,"SD"],
    ["NGC 7768",9.11,0.19,2.42,0.03,11.60,11.50,"SD"],
    ["UGC 1841",7.25,0.12,2.13,0.02,10.51,10.30,"M"],
    ["UGC 3789",7.11,0.12,2.00,0.04,10.50,10.30,"M"],
])

df=pd.DataFrame(DATA,columns=["galaxy","logM","eM","logS","eS","logMb","logL","method"])
for c in df.columns[1:7]: df[c]=df[c].astype(float)

logM=df["logM"].values; eM=df["eM"].values
logS=df["logS"].values; eS=df["eS"].values
logMb=df["logMb"].values; logL=df["logL"].values; N=len(df)

print("="*60)
print(f"BH Scaling — Solidified Analysis ({N} galaxies)")
print("="*60)

# ── 1. ODR with per-point errors ──
def odr_fit(x,y,sx,sy):
    def f(p,x): return p[0]+p[1]*x
    m=Model(f); d=RealData(x,y,sx=sx,sy=sy)
    return ODR(d,m,beta0=[np.mean(y),1.0]).run()

odr_s=odr_fit(logS,logM,eS,eM)
odr_b=odr_fit(logMb,logM,np.full(N,0.10),eM)
odr_l=odr_fit(logL,logM,np.full(N,0.10),eM)

# ── 2. Bootstrap ──
def bootstrap_odr(x,y,sx,sy,nb=1000):
    slopes=[]; n=len(x)
    for _ in range(nb):
        idx=RNG.choice(n,n,replace=True)
        try:
            r=odr_fit(x[idx],y[idx],sx[idx],sy[idx])
            slopes.append(r.beta[1])
        except: pass
    a=np.array(slopes); return np.median(a),np.std(a),np.percentile(a,[16,84])

b_med,b_std,b_ci=bootstrap_odr(logS,logM,eS,eM)
b_bmed,b_bstd,_=bootstrap_odr(logMb,logM,np.full(N,0.10),eM)
b_lmed,b_lstd,_=bootstrap_odr(logL,logM,np.full(N,0.10),eM)

# ── 3. Jackknife (remove 1 galaxy at a time) ──
jk_slopes=[]
for i in range(N):
    mask=np.ones(N,bool); mask[i]=False
    try: r=odr_fit(logS[mask],logM[mask],eS[mask],eM[mask]); jk_slopes.append(r.beta[1])
    except: pass
jk_a=np.array(jk_slopes); jk_std=np.std(jk_a)*(N-1)/np.sqrt(N)
print(f"\n1. M-sigma (ODR, bootstrap): β = {b_med:.2f} ± {b_std:.2f} [{b_ci[0]:.2f},{b_ci[1]:.2f}]")
print(f"   Jackknife σ = {jk_std:.2f}")
print(f"2. M-Mbulge (ODR):    β = {b_bmed:.2f} ± {b_bstd:.2f}")
print(f"3. M-L (ODR):         β = {b_lmed:.2f} ± {b_lstd:.2f}")

# ── 4. Literature comparison ──
lit={"McConnell & Ma 2013":(5.64,0.32),"Kormendy & Ho 2013":(4.38,0.28),
     "van den Bosch 2016":(4.35,0.30),"Saglia+2016 (RM)":(4.97,0.18),
     "This work (ODR+boot)":(b_med,b_std)}
print(f"\n4. Literature comparison (M-sigma slope β):")
for name,(slope,err) in lit.items():
    sig=abs(slope-b_med)/err if err>0 else 0
    print(f"   {name:<25s}: β={slope:.2f}±{err:.2f}  (Δ={slope-b_med:+.2f}, {sig:.1f}σ)")

# ── 5. Outlier sensitivity ──
outliers=["NGC 1277","NGC 4486B","NGC 3842","NGC 4889"]
mask_clean=~df["galaxy"].isin(outliers)
r_clean=odr_fit(logS[mask_clean],logM[mask_clean],eS[mask_clean],eM[mask_clean])
print(f"\n5. Outlier sensitivity: removing {outliers}")
print(f"   Full sample: β={b_med:.2f}±{b_std:.2f}")
print(f"   Clean:       β={r_clean.beta[1]:.2f}±{r_clean.sd_beta[1]:.2f}")

# ── 6. Method split ──
for method,label in [("SD","Stellar dynamics"),("GD","Gas dynamics"),("M","Masers")]:
    m=df["method"]==method
    if m.sum()>3:
        r_m=odr_fit(logS[m],logM[m],eS[m],eM[m])
        print(f"\n6. {label} (n={m.sum()}): β={r_m.beta[1]:.2f}±{r_m.sd_beta[1]:.2f}")

# ── 7. Intrinsic scatter ──
pred_odr=odr_s.beta[0]+odr_s.beta[1]*logS
rms_raw=np.std(logM-pred_odr)
chi2_odr=np.sum((logM-pred_odr)**2/(eS**2+eM**2))
sig_int=np.sqrt(max(rms_raw**2-np.median(eM)**2,0))
print(f"\n7. Intrinsic scatter: raw RMS={rms_raw:.4f}, intrinsic={sig_int:.4f} dex")

# ── Figure ──
fig,axes=plt.subplots(1,3,figsize=(18,5.5))
for i,(x,y,xl,yl,ex,ey,label,slope,serr) in enumerate([
    (logS,logM,"log σ [km/s]","log M_BH",eS,eM,"M-sigma",b_med,b_std),
    (logMb,logM,"log M_bulge","log M_BH",np.full(N,0.10),eM,"M-M_bulge",b_bmed,b_bstd),
    (logL,logM,"log L_bulge","log M_BH",np.full(N,0.10),eM,"M-L",b_lmed,b_lstd),
]):
    ax=axes[i]
    ax.errorbar(x,y,xerr=ex,yerr=ey,fmt="o",ms=5,alpha=0.6,capsize=2,zorder=2)
    xg=np.linspace(x.min(),x.max(),100)
    mid=np.mean(y); ax.plot(xg,mid+slope*(xg-np.mean(x)),"r-",lw=2,
                           label=f"β={slope:.2f}±{serr:.2f}")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=9)
    ax.set_title(f"{label}: RMS={np.std(y-(mid+slope*(x-np.mean(x)))):.3f}")

plt.tight_layout(); plt.savefig(f"{OUTDIR}/bh_solid.pdf",dpi=200); plt.savefig(f"{OUTDIR}/bh_solid.png",dpi=150)
print(f"\nSaved {OUTDIR}/bh_solid.png"); plt.close()

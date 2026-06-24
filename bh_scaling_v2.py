"""BH scaling v2: ODR fitting, error propagation, sample splits."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from scipy.odr import ODR, Model, RealData
import os, warnings; warnings.filterwarnings("ignore")

OUTDIR="analysis/bh_scaling"; os.makedirs(OUTDIR,exist_ok=True)
RNG=np.random.RandomState(42)

# Full catalog from v1
DATA = np.array([
    ["Milky Way",6.63,1.60,10.50,10.40], ["M31",8.18,2.05,11.00,10.90],
    ["M32",6.40,1.85,9.20,9.08], ["M59",8.63,2.34,11.30,11.10],
    ["M60",9.35,2.47,11.80,11.60], ["M81",7.81,2.02,10.90,10.70],
    ["M84",8.95,2.41,11.30,11.20], ["M87",9.79,2.53,12.00,11.90],
    ["M89",8.64,2.35,11.50,11.30], ["M104",8.81,2.36,11.60,11.40],
    ["M105",8.24,2.20,11.00,10.80], ["NGC221",6.40,1.85,9.20,9.08],
    ["NGC224",8.18,2.05,11.00,10.90], ["NGC821",7.74,2.19,11.10,10.90],
    ["NGC1023",7.61,2.18,10.80,10.60], ["NGC1068",6.93,2.02,10.70,10.60],
    ["NGC1194",7.89,2.02,10.60,10.40], ["NGC1271",9.00,2.44,11.30,11.20],
    ["NGC1277",9.98,2.51,11.60,11.40], ["NGC1300",7.15,2.00,10.50,10.30],
    ["NGC1316",8.30,2.34,11.40,11.20], ["NGC1332",8.99,2.47,11.50,11.30],
    ["NGC1399",8.85,2.51,11.70,11.60], ["NGC1407",9.70,2.43,11.70,11.60],
    ["NGC1550",9.07,2.41,11.20,11.10], ["NGC1600",9.78,2.51,12.00,11.90],
    ["NGC2549",7.15,2.16,10.60,10.40], ["NGC2748",7.62,1.93,10.30,10.20],
    ["NGC2778",7.15,2.02,10.10,10.00], ["NGC2787",7.60,2.04,10.30,10.20],
    ["NGC2960",7.08,2.00,10.60,10.40], ["NGC2974",8.23,2.32,11.00,10.80],
    ["NGC3079",6.30,2.02,10.60,10.50], ["NGC3115",8.95,2.37,11.10,10.90],
    ["NGC3227",7.67,2.08,10.80,10.60], ["NGC3245",8.36,2.24,10.90,10.70],
    ["NGC3377",8.08,2.02,10.30,10.20], ["NGC3379",8.08,2.19,10.90,10.70],
    ["NGC3384",7.23,2.02,10.50,10.30], ["NGC3414",8.40,2.35,11.20,11.00],
    ["NGC3489",6.78,1.89,10.50,10.30], ["NGC3585",8.40,2.27,11.30,11.10],
    ["NGC3607",8.15,2.32,11.10,10.90], ["NGC3608",8.66,2.19,10.80,10.60],
    ["NGC3842",9.97,2.46,12.00,11.80], ["NGC3998",8.91,2.42,10.80,10.60],
    ["NGC4026",8.26,2.02,10.90,10.70], ["NGC4151",7.62,1.99,10.70,10.60],
    ["NGC4258",7.60,1.93,10.50,10.30], ["NGC4261",8.73,2.43,11.50,11.40],
    ["NGC4291",8.96,2.42,11.20,11.10], ["NGC4342",8.64,2.34,10.60,10.50],
    ["NGC4374",9.10,2.44,11.50,11.30], ["NGC4382",7.15,2.02,11.30,11.10],
    ["NGC4459",7.86,2.01,10.60,10.40], ["NGC4473",7.95,2.04,10.60,10.40],
    ["NGC4486",7.63,2.17,11.60,11.50], ["NGC4486A",7.15,1.88,9.80,9.70],
    ["NGC4486B",8.76,2.02,10.50,10.40], ["NGC4526",7.80,2.17,11.00,10.80],
    ["NGC4564",7.65,2.09,10.60,10.40], ["NGC4594",8.82,2.36,11.50,11.30],
    ["NGC4649",9.68,2.49,11.70,11.50], ["NGC4697",8.27,2.02,10.80,10.60],
    ["NGC4736",6.76,1.89,10.30,10.20], ["NGC4742",7.15,1.94,10.30,10.20],
    ["NGC4751",8.23,2.22,11.00,10.80], ["NGC4762",6.76,2.00,10.90,10.80],
    ["NGC4889",9.91,2.54,12.20,12.00], ["NGC4945",6.04,1.93,9.00,8.90],
    ["NGC5077",8.85,2.38,11.40,11.20], ["NGC5128",7.70,2.02,11.10,10.90],
    ["NGC5252",7.57,2.04,10.70,10.50], ["NGC5328",8.51,2.41,11.10,11.00],
    ["NGC5419",9.22,2.54,11.90,11.80], ["NGC5490",9.15,2.42,11.70,11.50],
    ["NGC5516",8.57,2.43,11.30,11.20], ["NGC5576",8.20,2.22,10.90,10.80],
    ["NGC5813",8.86,2.37,11.60,11.50], ["NGC5845",8.53,2.35,10.50,10.30],
    ["NGC5846",9.08,2.37,11.70,11.50], ["NGC6086",9.15,2.43,11.30,11.10],
    ["NGC6251",8.81,2.42,11.40,11.30], ["NGC6323",7.78,2.07,10.70,10.60],
    ["NGC7052",8.58,2.44,11.30,11.10], ["NGC7457",6.45,1.72,9.80,9.70],
    ["NGC7582",7.76,2.02,10.80,10.60], ["NGC7619",8.99,2.45,11.70,11.60],
    ["NGC7768",9.11,2.42,11.60,11.50], ["CygA",9.40,2.59,11.90,11.70],
    ["IC1459",9.40,2.51,11.50,11.40], ["IC4296",9.20,2.51,11.60,11.50],
])
df=pd.DataFrame(DATA,columns=["galaxy","logM","logS","logMb","logL"])
for c in df.columns[1:]: df[c]=df[c].astype(float)

logM=df["logM"].values; logS=df["logS"].values; logMb=df["logMb"].values; logL=df["logL"].values
N=len(df)

# ── ODR fitting (errors in both X and Y) ──
def odr_fit(x,y,ex=0.05,ey=0.10):
    """Orthogonal distance regression with errors in both variables."""
    def f(params,x): return params[0]+params[1]*x
    model=Model(f)
    d=RealData(x,y,sx=np.full(len(x),ex),sy=np.full(len(y),ey))
    odr=ODR(d,model,beta0=[np.mean(y),1.0])
    return odr.run()

odr_s=odr_fit(logS,logM,0.05,0.10)
odr_b=odr_fit(logMb,logM,0.10,0.10)
odr_l=odr_fit(logL,logM,0.10,0.10)

print("ODR (errors in both variables):")
print(f"  M-sigma:    slope={odr_s.beta[1]:.2f}±{odr_s.sd_beta[1]:.2f}")
print(f"  M-M_bulge:  slope={odr_b.beta[1]:.2f}±{odr_b.sd_beta[1]:.2f}")
print(f"  M-L:        slope={odr_l.beta[1]:.2f}±{odr_l.sd_beta[1]:.2f}")

# ── Approach 3: OLS with measurement errors ──
# Add typical measurement uncertainties: 0.05 dex in log sigma, 0.10 dex in log M_BH
sig_int=0.44
for label,x,y in [("M-sigma",logS,logM),("M-M_bulge",logMb,logM),("M-L",logL,logM)]:
    ex=0.05; ey=0.10
    tot_err=np.sqrt(ex**2+ey**2+sig_int**2)
    p,_=curve_fit(lambda x,a,b:a+b*x,x,y,sigma=np.full(N,tot_err),absolute_sigma=True)
    # Bootstrap
    bts=[]
    for _ in range(500):
        idx=RNG.choice(N,N,replace=True)
        try:
            pt,_=curve_fit(lambda x,a,b:a+b*x,x[idx],y[idx],sigma=np.full(N,tot_err),absolute_sigma=True,maxfev=10000)
            bts.append(pt[1])
        except: pass
    ba=np.array(bts)
    chi2=np.sum((y-(p[0]+p[1]*x))**2/tot_err**2)
    print(f"  {label}: slope={p[1]:.2f}±{np.std(ba):.2f} (χ²={chi2:.0f}, χ²_red={chi2/(N-2):.2f})")

# ── Subsample: high sigma only ──
high=logS>2.2
if high.sum()>15:
    p_h,_=curve_fit(lambda x,a,b:a+b*x,logS[high],logM[high],sigma=np.full(high.sum(),0.44),absolute_sigma=True)
    print(f"\nHigh-σ (σ>160 km/s, n={high.sum()}): slope={p_h[1]:.2f}")
# Low sigma
low=logS<=2.2
if low.sum()>15:
    p_l,_=curve_fit(lambda x,a,b:a+b*x,logS[low],logM[low],sigma=np.full(low.sum(),0.44),absolute_sigma=True)
    print(f"Low-σ (σ≤160 km/s, n={low.sum()}):  slope={p_l[1]:.2f}")

# ── Figure ──
fig,ax=plt.subplots(1,3,figsize=(18,5.5))
for i,(label,x,y,xl,yl) in enumerate([
    ("M-sigma",logS,logM,"log σ [km/s]","log M_BH"),
    ("M-M_bulge",logMb,logM,"log M_bulge","log M_BH"),
    ("M-L",logL,logM,"log L_bulge","log M_BH"),
]):
    ax[i].scatter(x,y,s=20,alpha=0.7,edgecolors="k",lw=0.5)
    xg=np.linspace(x.min(),x.max(),100)
    # OLS
    p_ols=np.polyfit(x,y,1)
    ax[i].plot(xg,p_ols[0]*xg+p_ols[1],"b-",lw=1.5,label=f"OLS: {p_ols[0]:.2f}")
    # ODR
    if i==0: ax[i].plot(xg,odr_s.beta[0]+odr_s.beta[1]*xg,"r--",lw=1.5,label=f"ODR: {odr_s.beta[1]:.2f}")
    elif i==1: ax[i].plot(xg,odr_b.beta[0]+odr_b.beta[1]*xg,"r--",lw=1.5,label=f"ODR: {odr_b.beta[1]:.2f}")
    else: ax[i].plot(xg,odr_l.beta[0]+odr_l.beta[1]*xg,"r--",lw=1.5,label=f"ODR: {odr_l.beta[1]:.2f}")
    ax[i].set_xlabel(xl); ax[i].set_ylabel(yl); ax[i].legend(fontsize=8)
    ax[i].set_title(f"{label}: RMS={np.std(y-p_ols[0]*x-p_ols[1]):.3f}")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/bh_v2.pdf",dpi=200); plt.savefig(f"{OUTDIR}/bh_v2.png",dpi=150)
plt.close(); print(f"\nSaved {OUTDIR}/bh_v2.png")

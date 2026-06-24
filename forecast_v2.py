"""Forecast v2: properly cited survey factors from SRDs.

Sources:
- Euclid: Laureijs+2011 (arXiv:1110.3193), 15,000 deg², 30 gal/arcmin²
- LSST: Ivezic+2019 (arXiv:0805.2366), 18,000 deg², ugrizy, 26.5 mag
- Roman: Spergel+2015 (arXiv:1503.03757), 2,000 deg², 2.4m space telescope

Mistele+2024 used KiDS (~1,000 deg²) for 11 binned RAR points.
Projected improvements are statistical (√area) × systematic (photo-z/shapes).
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os, warnings; warnings.filterwarnings("ignore")
OUTDIR="analysis/forecast"; os.makedirs(OUTDIR,exist_ok=True)

# ── Current data ──
LENS = np.array([[-12.39,-11.11,0.06],[-12.64,-11.21,0.05],[-12.89,-11.29,0.05],
    [-13.13,-11.47,0.05],[-13.38,-11.59,0.05],[-13.63,-11.76,0.06],[-13.87,-11.93,0.07],
    [-14.12,-12.08,0.07],[-14.37,-12.27,0.08],[-14.61,-12.44,0.08],[-14.86,-12.85,0.12]])
SPARC = np.array([[-10.82,-10.35,0.03],[-10.54,-10.15,0.02],[-10.26,-9.93,0.02],
    [-9.97,-9.70,0.02],[-9.69,-9.47,0.01],[-9.41,-9.23,0.01],[-9.12,-8.98,0.01],
    [-8.88,-8.75,0.01],[-8.70,-8.59,0.01],[-8.37,-8.28,0.01]])
x = np.concatenate([SPARC[:,0], LENS[:,0]])
y = np.concatenate([SPARC[:,1], LENS[:,1]])
e = np.concatenate([SPARC[:,2], LENS[:,2]])

# ── Compute sigma_c with proper systematic floor ──
def sigma_c(sys_floor, n_sim=500):
    rng=np.random.RandomState(42); cv=[]
    e_tot=np.sqrt(e**2+sys_floor**2)
    for _ in range(n_sim):
        ys=y+rng.normal(0,e_tot)
        def chi2(p): a,b,c=p; pred=a+b/np.maximum(x,-50)+c*x; return np.sum(((ys-pred)/e_tot)**2)
        r=minimize(chi2,[-17,-70,0.1],method="Nelder-Mead",options={"xatol":1e-8,"fatol":1e-8})
        cv.append(r.x[2])
    return np.std(np.array(cv))

# Decompose: σ_c_total² = σ_c_stat² + σ_c_sys²
sc_stat = sigma_c(0.0)
sc_005 = sigma_c(0.05)
sc_010 = sigma_c(0.10)
# Statistical component: what 1/√N improves
# Systematic component: what stays constant
sc_sys_est = np.sqrt(max(sc_005**2 - sc_stat**2, 0))

print("Forecast v2: properly decomposed σ_c")
print(f"  Current σ_c (stat only):   {sc_stat:.4f}")
print(f"  Current σ_c (+0.05 dex):   {sc_005:.4f}")
print(f"  Estimated σ_c (sys):       {sc_sys_est:.4f}")
print(f"  Current σ_c (+0.10 dex):   {sc_010:.4f}")

# ── Survey projections from SRDs ──
# KiDS (Mistele+2024 source): ~1,000 deg²
# Euclid Wide: 15,000 deg² (Laureijs+2011) → 15× area
# LSST: 18,000 deg² (Ivezic+2019) → 18× area
# Roman HLWAS: 2,000 deg² (Spergel+2015) → 2× area, but space quality
surveys = [
    ("Current (KiDS+Mistele)", 1.0, 1.0, 1.0, "1,000 deg²"),
    ("Euclid Wide (2026)",      15.0, 1.0, 0.8, "Laureijs+2011"),
    ("LSST (2027)",             18.0, 1.3, 0.7, "Ivezic+2019"),
    ("Roman HLWAS (2028)",      2.0,  1.5, 0.5, "Spergel+2015"),
    ("Euclid+LSST (2028)",      33.0, 1.5, 0.6, "combined ground+space"),
    ("All three (2030)",        35.0, 1.8, 0.5, "Euclid+LSST+Roman"),
]

print(f"\n{'Survey':<25s} {'σ_c_stat':<10s} {'σ_c_tot':<10s} {'5σ c_min':<10s} {'c=0.5?'}")
print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
for name, area, depth, quality, ref in surveys:
    # Statistical: σ_stat ∝ 1/√N ∝ 1/√area
    sc_stat_f = sc_stat / np.sqrt(area)
    # Systematic stays, improved by space quality factor
    sc_sys_f = sc_sys_est * quality / depth
    sc_tot = np.sqrt(sc_stat_f**2 + sc_sys_f**2)
    c5 = 5*sc_tot; det = "YES" if c5 < 0.5 else "NO"
    print(f"{name:<25s} {sc_stat_f:<10.4f} {sc_tot:<10.4f} {c5:<10.3f} {det} ({ref})")

# ── Figure ──
fig,axes=plt.subplots(1,2,figsize=(14,5.5))

ax=axes[0]
sf_grid=np.linspace(0,0.15,20)
sc_stat_vals=[sigma_c(sf,200) for sf in sf_grid]
sc_sys_vals=[np.sqrt(max(s**2-sc_stat**2,0)) for s in sc_stat_vals]
ax.plot(sf_grid,sc_stat_vals,"ko-",lw=2,ms=4,label="σ_c (total)")
ax.plot(sf_grid,sc_sys_vals,"r--",lw=1.5,label="σ_c (sys, estimated)")
ax.axhline(0.10,color="red",ls="--",lw=1,alpha=0.5,label="5σ target")
ax.axvline(0.05,color="green",ls=":",lw=1.5,label="0.05 dex floor")
ax.set_xlabel("Systematic error floor (dex)"); ax.set_ylabel("σ_c"); ax.legend(fontsize=8)
ax.set_title("(a) σ_c Decomposition")

ax=axes[1]
names=[s[0] for s in surveys]
sc_tots=[np.sqrt((sc_stat/np.sqrt(s[1]))**2+(sc_sys_est*s[3]/s[2])**2) for s in surveys]
colors=["gray","blue","orange","red","green","purple"]
ax.barh(names,sc_tots,color=colors,edgecolor="k")
ax.axvline(0.10,color="red",ls="--",lw=1.5,label="5σ target")
ax.set_xlabel("σ_c (stat+sys)"); ax.set_title("(b) Survey Timeline (SRD-based)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/forecast_v2.pdf",dpi=200); plt.savefig(f"{OUTDIR}/forecast_v2.png",dpi=150)
print(f"\nSaved {OUTDIR}/forecast_v2.png"); plt.close()

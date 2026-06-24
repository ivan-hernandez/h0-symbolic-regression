"""Forecast: When will surveys definitively test the CPX5 RAR form?

CPX5: log g_obs = a + b/log_gbar (c=0, no sqrt asymptote)
MOND: c=0.5 (sqrt asymptote emerges at low g_bar)

σ_c = current uncertainty on the slope parameter c.
5σ detection of c=0 requires σ_c < 0.10.
This script computes σ_c for current + future surveys.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os, warnings; warnings.filterwarnings("ignore")
OUTDIR="analysis/forecast"; os.makedirs(OUTDIR,exist_ok=True)

# ── Current data (Mistele+2024 lensing + SPARC binned) ──
LENSING = np.array([[-12.39,-11.11,0.06],[-12.64,-11.21,0.05],[-12.89,-11.29,0.05],
    [-13.13,-11.47,0.05],[-13.38,-11.59,0.05],[-13.63,-11.76,0.06],[-13.87,-11.93,0.07],
    [-14.12,-12.08,0.07],[-14.37,-12.27,0.08],[-14.61,-12.44,0.08],[-14.86,-12.85,0.12]])
SPARC_B = np.array([[-10.82,-10.35,0.03],[-10.54,-10.15,0.02],[-10.26,-9.93,0.02],
    [-9.97,-9.70,0.02],[-9.69,-9.47,0.01],[-9.41,-9.23,0.01],[-9.12,-8.98,0.01],
    [-8.88,-8.75,0.01],[-8.70,-8.59,0.01],[-8.37,-8.28,0.01]])
x_all = np.concatenate([SPARC_B[:,0], LENSING[:,0]])
y_all = np.concatenate([SPARC_B[:,1], LENSING[:,1]])
e_stat = np.concatenate([SPARC_B[:,2], LENSING[:,2]])

# ── Compute sigma_c via Monte Carlo ──
def sigma_c_mc(sys_floor=0.05, n_sim=500):
    e_tot = np.sqrt(e_stat**2 + sys_floor**2)
    c_vals = []; rng = np.random.RandomState(42)
    for _ in range(n_sim):
        y_sim = y_all + rng.normal(0, e_tot)
        def chi2(p):
            a,b,c=p
            pred=a+b/np.maximum(x_all,-50)+c*x_all
            return np.sum(((y_sim-pred)/e_tot)**2)
        r = minimize(chi2, [-17,-70,0.1], method="Nelder-Mead", options={"xatol":1e-8,"fatol":1e-8})
        c_vals.append(r.x[2])
    return np.std(np.array(c_vals))

print("Forecast: CPX5 vs MOND asymptote (5σ detection of c=0)")
print("="*60)
print(f"\nSystematic floor sensitivity:")
for sf in [0.0, 0.02, 0.05, 0.10, 0.15]:
    sc = sigma_c_mc(sf)
    det = "YES" if 5*sc < 0.5 else "NO"
    print(f"  σ_sys={sf:.2f} dex: σ_c={sc:.4f}, 5σ_threshold={5*sc:.3f} → detect c=0.5? {det}")

# ── Survey projections ──
sigma_c_now = sigma_c_mc(0.05)  # realistic floor: 0.05 dex
print(f"\nBaseline (σ_sys=0.05 dex): σ_c = {sigma_c_now:.4f}")

surveys = {
    "Mistele+2024 (current)":   {"fp":1,   "fr":1.0, "fe":1.0},
    "Euclid Wide (2025)":       {"fp":3,   "fr":1.1, "fe":0.7},
    "Rubin/LSST (2025)":        {"fp":10,  "fr":1.3, "fe":0.6},
    "Roman HLWAS (2027)":       {"fp":5,   "fr":1.5, "fe":0.4},
    "Combined (2030)":          {"fp":30,  "fr":2.0, "fe":0.3},
}
print(f"\n{'Survey':<25s} {'σ_c':<10s} {'N_eff/range':<15s} {'5σ c_min':<12s} {'c=0.5?'}")
print(f"{'-'*25} {'-'*10} {'-'*15} {'-'*12} {'-'*8}")
for name, s in surveys.items():
    sc = sigma_c_now / np.sqrt(s["fp"]) / s["fr"] * s["fe"]
    c5 = 5*sc; det = "YES" if c5 < 0.5 else "NO"
    n_range = f"{s['fp']}×/{s['fr']:.1f}×"
    print(f"{name:<25s} {sc:<10.4f} {n_range:<15s} {c5:<12.3f} {det}")

# ── Figure ──
fig,axes=plt.subplots(1,2,figsize=(14,5.5))

ax=axes[0]
sf_grid=np.linspace(0,0.15,20); sc_vals=[sigma_c_mc(sf,200) for sf in sf_grid]
ax.plot(sf_grid,sc_vals,"ko-",lw=2,ms=4)
ax.axhline(0.10,color="red",ls="--",lw=1.5,label="σ_c=0.10 (5σ target)")
ax.axvline(0.05,color="green",ls=":",lw=1.5,label="Realistic floor 0.05 dex")
ax.set_xlabel("Systematic error floor (dex)"); ax.set_ylabel("σ_c (slope uncertainty)")
ax.set_title("(a) σ_c vs Systematic Floor"); ax.legend(fontsize=8)

ax=axes[1]
names=list(surveys.keys())
sc_disp=[sigma_c_now/np.sqrt(s["fp"])/s["fr"]*s["fe"] for s in surveys.values()]
colors=["gray","blue","orange","red","purple"]
ax.barh(names,sc_disp,color=colors,edgecolor="k")
ax.axvline(0.10,color="red",ls="--",lw=1.5,label="5σ target")
ax.set_xlabel("σ_c"); ax.set_title(f"(b) σ_c Timeline (σ_sys=0.05 dex)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/forecast_paper.pdf",dpi=200); plt.savefig(f"{OUTDIR}/forecast_paper.png",dpi=150)
print(f"\nSaved {OUTDIR}/forecast_paper.png"); plt.close()

"""Phase 3 fix: Systematic floor in survey forecast. Monte Carlo based."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

OUTDIR = "analysis/phase3"

LENSING_DATA = np.array([
    [-12.39, -11.11, 0.06], [-12.64, -11.21, 0.05],
    [-12.89, -11.29, 0.05], [-13.13, -11.47, 0.05],
    [-13.38, -11.59, 0.05], [-13.63, -11.76, 0.06],
    [-13.87, -11.93, 0.07], [-14.12, -12.08, 0.07],
    [-14.37, -12.27, 0.08], [-14.61, -12.44, 0.08],
    [-14.86, -12.85, 0.12],
])
SPARC_BINNED = np.array([
    [-10.82, -10.35, 0.03], [-10.54, -10.15, 0.02],
    [-10.26, -9.93, 0.02], [-9.97, -9.70, 0.02],
    [-9.69, -9.47, 0.01], [-9.41, -9.23, 0.01],
    [-9.12, -8.98, 0.01], [-8.88, -8.75, 0.01],
    [-8.70, -8.59, 0.01], [-8.37, -8.28, 0.01],
])

x_all = np.concatenate([SPARC_BINNED[:,0], LENSING_DATA[:,0]])
y_all = np.concatenate([SPARC_BINNED[:,1], LENSING_DATA[:,1]])
err_stat = np.concatenate([SPARC_BINNED[:,2], LENSING_DATA[:,2]])

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 60)
print("Phase 3 Fix: Systematic Floor Forecast (Monte Carlo)")
print("=" * 60)

def mc_sigma_c(err_total, n_sim=1000):
    c_vals = []
    rng = np.random.RandomState(42)
    for _ in range(n_sim):
        y_sim = y_all + rng.normal(0, err_total)
        def chi2(params):
            a, b, c = params
            pred = a + b/np.maximum(x_all, -50) + c*x_all
            return np.sum(((y_sim - pred)/err_total)**2)
        r = minimize(chi2, [-17, -70, 0.1], method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-8})
        c_vals.append(r.x[2])
    return np.nanstd(c_vals)

sys_floors = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
sigma_c_vals = []

print(f"\n  {'σ_sys':<12s} {'σ_c (MC)':<12s} {'5σ c_min':<12s} {'Detect 0.5?'}")
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*15}")
for sf in sys_floors:
    err_tot = np.sqrt(err_stat**2 + sf**2)
    sc = mc_sigma_c(err_tot, n_sim=500)
    sigma_c_vals.append(sc)
    c5 = 5*sc
    det = "YES" if c5 < 0.5 else "NO"
    print(f"  {sf:<12.2f} {sc:<12.4f} {c5:<12.3f} {det}")

# Forecast with 0.05 dex floor
surveys = {
    "Euclid (2025)":       {"fp": 3, "fr": 1.0, "fe": 0.7},
    "Rubin/LSST (2025)":   {"fp": 10, "fr": 1.2, "fe": 0.6},
    "Roman (2027)":        {"fp": 5, "fr": 1.5, "fe": 0.4},
    "Combined (2030)":     {"fp": 30, "fr": 2.0, "fe": 0.3},
}
base_sc = sigma_c_vals[2]  # 0.05 dex floor
print(f"\n  With σ_sys = 0.05 dex floor, current σ_c = {base_sc:.4f}")
print(f"  {'Survey':<22s} {'σ_c':<10s} {'5σ c_min':<12s} {'Detect 0.5?'}")
for name, f in surveys.items():
    sc = base_sc / np.sqrt(f["fp"]) / f["fr"] * f["fe"]
    c5 = 5*sc
    det = "YES" if c5 < 0.5 else "NO"
    print(f"  {name:<22s} {sc:<10.4f} {c5:<12.3f} {det}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(sys_floors, sigma_c_vals, "ko-", lw=2, ms=6)
ax.axhline(0.10, color="red", ls="--", lw=1.5, label="σ_c=0.10 (5σ target)")
ax.axvline(0.05, color="green", ls=":", lw=1.5, label="Realistic floor 0.05 dex")
ax.set_xlabel("Systematic error floor (dex)")
ax.set_ylabel("σ_c")
ax.set_title("(a) σ_c vs Systematic Floor (Monte Carlo)")
ax.legend(fontsize=8)

ax = axes[1]
names = ["Current", "Euclid", "Rubin", "Roman", "2030"]
sc_vals = [base_sc]
for name, f in surveys.items():
    sc_vals.append(base_sc / np.sqrt(f["fp"]) / f["fr"] * f["fe"])
ax.barh(names, sc_vals, color="steelblue", edgecolor="k")
ax.axvline(0.10, color="red", ls="--", lw=1.5, label="5σ threshold")
ax.set_xlabel("σ_c")
ax.set_title(f"(b) Forecast (σ_sys=0.05 dex, σ_c_current={base_sc:.3f})")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/forecast_with_systematics.pdf", dpi=200)
plt.savefig(f"{OUTDIR}/forecast_with_systematics.png", dpi=150)
print(f"\n  Saved {OUTDIR}/forecast_with_systematics.png")
plt.close()

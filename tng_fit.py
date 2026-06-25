"""TNG100 real simulation RAR: download group catalog, fit CPX5.

Uses TNG public API with user's API key. Downloads group catalog
data, computes g_bar and g_obs from subhalo properties, and fits
CPX5 to compare with SPARC observations.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import urllib.request, json, os, time, warnings
warnings.filterwarnings("ignore")

API_KEY = os.environ.get("TNG_API_KEY", "")
BASE = "https://www.tng-project.org/api/"
OUTDIR = "analysis/tng"
os.makedirs(OUTDIR, exist_ok=True)

G_SI = 6.6743e-11; Msun_kg = 1.989e30; kpc_m = 3.0857e19

def api_get(url, params=None):
    """Make authenticated TNG API request."""
    headers = {"api-key": API_KEY}
    full_url = BASE + url
    if params:
        full_url += "?" + "&".join(f"{k}={v}" for k,v in params.items())
    req = urllib.request.Request(full_url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def get_subhalo_page(sim="TNG100-1", snap=99, limit=100, offset=0):
    """Get one page of subhalo data."""
    url = f"{sim}/snapshots/{snap}/subhalos/"
    return api_get(url, {"limit": str(limit), "offset": str(offset)})

# Step 1: Get total count
print("Checking TNG catalog...")
first_page = get_subhalo_page(limit=1)
count = first_page["count"]
print(f"  Total subhalos: {count}")

# Step 2: Download in pages
subhalos = []
limit = 100
n_pages = min(count // limit + 1, 50)  # limit to 5000 galaxies
print(f"  Downloading {n_pages} pages ({n_pages*limit} subhalos)...")

for page in range(n_pages):
    offset = page * limit
    data = get_subhalo_page(limit=limit, offset=offset)
    for sh in data["results"]:
        subhalos.append({
            "mass": sh["mass"],           # total mass (10^10 Msun/h)
            "vmax": sh["vmax"],           # max circular velocity (km/s)
            "rhalf": sh["halfmassrad"],   # half-mass radius (ckpc/h)
            "mass_stars": sh["mass_stars"] if sh["mass_stars"]>0 else 0,
            "mass_gas": sh["mass_gas"] if sh["mass_gas"]>0 else 0,
        })
    if (page+1) % 10 == 0:
        print(f"    ... {offset+limit}/{min(count, n_pages*limit)}")

df = pd.DataFrame(subhalos)
n = len(df)
print(f"  Downloaded {n} subhalos")

# Step 3: Compute RAR quantities
h = 0.6774  # TNG cosmology
df["mass_tot"] = df["mass"] * 1e10 / h  # Msun
df["mass_bar"] = (df["mass_stars"] + df["mass_gas"]) * 1e10 / h  # Msun
df["R_kpc"] = df["rhalf"] / h  # physical kpc
df["V_max"] = df["vmax"]

# Filter: reasonable galaxies (M_bar > 10^6 Msun, Vmax > 10 km/s)
df = df[(df["mass_bar"] > 1e6) & (df["V_max"] > 10) & (df["R_kpc"] > 0.1)]
print(f"  After quality cuts: {len(df)} subhalos")

# g_bar = G * M_bar / R²
df["g_bar"] = G_SI * df["mass_bar"] * Msun_kg / (df["R_kpc"] * kpc_m)**2
# g_obs = V_max² / R
df["g_obs"] = (df["V_max"] * 1000)**2 / (df["R_kpc"] * kpc_m)

valid = (df["g_bar"] > 1e-14) & (df["g_obs"] > 1e-14)
df = df[valid]
print(f"  After physics cuts: {len(df)} subhalos")

log_gbar = np.log10(df["g_bar"])
log_gobs = np.log10(df["g_obs"])
print(f"  log g_bar range: [{log_gbar.min():.2f}, {log_gbar.max():.2f}]")
print(f"  log g_obs range: [{log_gobs.min():.2f}, {log_gobs.max():.2f}]")

# Step 4: Fit CPX5
def cpx5_log(x, a, b):
    return a + b / np.maximum(x, -50)

popt, pcov = curve_fit(cpx5_log, log_gbar, log_gobs, p0=[-17, -70], maxfev=10000)
perr = np.sqrt(np.diag(pcov))
pred = cpx5_log(log_gbar, *popt)
rms = np.sqrt(np.mean((log_gobs - pred)**2))
print(f"\n  TNG100 CPX5 fit:")
print(f"    a = {popt[0]:.2f} ± {perr[0]:.2f}")
print(f"    b = {popt[1]:.2f} ± {perr[1]:.2f}")
print(f"    RMS = {rms:.4f} dex")

# Compare with SPARC
sparc_a, sparc_b = -17.06, -72.71
da = popt[0] - sparc_a
db = popt[1] - sparc_b
d = np.sqrt(da**2 + (db/10)**2)
print(f"\n  SPARC reference: a={sparc_a}, b={sparc_b}")
print(f"    Δa = {da:+.2f}, Δb = {db:+.0f}")
print(f"    Distance in CPX5 space: {d:.2f}")

# Compare with Phase 3 synthetic values
SYNTHETIC = {
    "EAGLE": (-16.32, -66.72),
    "IllustrisTNG": (-16.48, -67.58),
    "FIRE-2": (-16.88, -71.13),
    "Baryonification": (-16.91, -71.51),
}
print(f"\n  Synthetic TNG (Phase 3): a={SYNTHETIC['IllustrisTNG'][0]:.2f}, b={SYNTHETIC['IllustrisTNG'][1]:.2f}")
for sim, (sa, sb) in SYNTHETIC.items():
    sd = np.sqrt((popt[0]-sa)**2 + ((popt[1]-sb)/10)**2)
    print(f"    Δ from {sim}: d={sd:.2f}")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
hb = ax.hexbin(log_gbar, log_gobs, gridsize=60, cmap="Blues", mincnt=1, bins="log")
x_grid = np.linspace(log_gbar.min(), log_gbar.max(), 200)
ax.plot(x_grid, cpx5_log(x_grid, *popt), "r-", lw=2.5,
        label=f"TNG100: a={popt[0]:.2f}, b={popt[1]:.0f}")
ax.plot(x_grid, cpx5_log(x_grid, sparc_a, sparc_b), "k--", lw=2, alpha=0.7,
        label=f"SPARC: a={sparc_a}, b={sparc_b}")
ax.plot(x_grid, x_grid, "k:", lw=0.5, alpha=0.3)
plt.colorbar(hb, ax=ax, label="Subhalos per bin")
ax.set_xlabel("log g_bar [m/s²]"); ax.set_ylabel("log g_obs [m/s²]")
ax.legend(fontsize=8); ax.set_title(f"(a) TNG100 RAR ({len(df)} subhalos, RMS={rms:.3f})")

ax = axes[1]
resid = log_gobs - cpx5_log(log_gbar, sparc_a, sparc_b)
hb2 = ax.hexbin(log_gbar, resid, gridsize=50, cmap="RdBu_r", mincnt=1)
ax.axhline(0, color="k", ls="--", lw=0.5)
ax.set_xlabel("log g_bar [m/s²]")
ax.set_ylabel("Residual from SPARC CPX5 (dex)")
ax.set_title(f"(b) TNG100 − SPARC CPX5 (mean={np.mean(resid):+.3f})")
plt.colorbar(hb2, ax=ax, label="Count")

plt.tight_layout(); plt.savefig(f"{OUTDIR}/tng100_rar.pdf", dpi=200); plt.savefig(f"{OUTDIR}/tng100_rar.png", dpi=150)
print(f"\n  Saved {OUTDIR}/tng100_rar.png"); plt.close()

# Save
df[["mass_tot","mass_bar","V_max","R_kpc","g_bar","g_obs"]].to_csv(f"{OUTDIR}/tng100_rar.csv", index=False)
print(f"  Saved {OUTDIR}/tng100_rar.csv")

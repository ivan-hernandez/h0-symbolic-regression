"""Fit CPX5 to real digitized simulation RAR data from Desmond+2023."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

OUTDIR = "analysis/phase3/simulation_data"
os.makedirs(OUTDIR, exist_ok=True)

def cpx5_log(x, a, b):
    return a + b / np.maximum(x, -50)

# Load digitized data — format: "x, y" per line
with open(os.path.expanduser("~/Downloads/Default Dataset (1).csv")) as f:
    lines = f.readlines()

points = []
for line in lines[1:]:  # skip header if present
    line = line.strip()
    if not line:
        continue
    parts = line.replace('"', '').split(",")
    if len(parts) >= 2:
        try:
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            points.append([x, y])
        except ValueError:
            pass

data = np.array(points)
x_raw, y_raw = data[:, 0], data[:, 1]
print(f"Parsed {len(data)} points")

# Clean: remove obvious outliers (>2σ from running median)
from scipy.ndimage import median_filter
for _ in range(2):
    y_smooth = median_filter(y_raw, size=5)
    resid = np.abs(y_raw - y_smooth)
    mask = resid < 3 * np.std(resid)
    x_raw, y_raw = x_raw[mask], y_raw[mask]

print(f"After outlier removal: {len(x_raw)} points")

# Sort
order = np.argsort(x_raw)
x, y = x_raw[order], y_raw[order]

# Fit CPX5
popt, pcov = curve_fit(cpx5_log, x, y, p0=[-17, -70], maxfev=10000)
a, b = popt
perr = np.sqrt(np.diag(pcov))
print(f"\nCPX5 fit to Desmond+2023 digitized data:")
print(f"  a = {a:.2f} ± {perr[0]:.2f}")
print(f"  b = {b:.2f} ± {perr[1]:.2f}")

# Compare with our synthetic simulation values
SIM_RESULTS = {
    "EAGLE":         {"a": -16.32, "b": -66.72},
    "IllustrisTNG":  {"a": -16.48, "b": -67.58},
    "FIRE-2":        {"a": -16.88, "b": -71.13},
    "MassiveBlack-II": {"a": -18.72, "b": -88.90},
    "Baryonification": {"a": -16.91, "b": -71.51},
}
SIM_COLORS = {
    "EAGLE": "orange", "IllustrisTNG": "green", "FIRE-2": "red",
    "MassiveBlack-II": "purple", "Baryonification": "brown",
}
print(f"\nComparison with synthetic simulation CPX5 parameters:")
for sim, params in SIM_RESULTS.items():
    da = a - params["a"]
    db = b - params["b"]
    d = np.sqrt(da**2 + (db/10)**2)
    print(f"  {sim:<20s}: a={params['a']:.2f} b={params['b']:.2f}  d={d:.2f}")

# Find closest simulation
distances = {sim: np.sqrt((a-params['a'])**2 + ((b-params['b'])/10)**2)
             for sim, params in SIM_RESULTS.items()}
closest = min(distances, key=distances.get)
print(f"\n  Closest match: {closest} (d={distances[closest]:.2f})")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
x_grid = np.linspace(-13.5, -8, 300)
ax.scatter(x, y, s=5, c="blue", alpha=0.8, label="Desmond+2023 (digitized)")
ax.plot(x_grid, cpx5_log(x_grid, a, b), "b-", lw=2, label=f"CPX5 fit: a={a:.2f}, b={b:.2f}")
# SPARC data for reference
ax.plot(x_grid, cpx5_log(x_grid, -17.06, -72.71), "k--", lw=1, alpha=0.5, label="SPARC CPX5")
ax.set_xlabel("log g_bar [m/s²]")
ax.set_ylabel("log g_obs [m/s²]")
ax.set_title("(a) Desmond+2023 RAR — Real Digitized Data")
ax.legend(fontsize=8)
ax.set_xlim(-13.5, -8)
ax.set_ylim(-13, -8)

ax = axes[1]
# CPX5 parameter space with digitized data
for sim, params in SIM_RESULTS.items():
    ax.scatter(params["a"], params["b"], c=SIM_COLORS[sim], s=100,
               edgecolors="k", linewidth=0.8, label=sim)
    ax.annotate(sim.replace(" ", "\n")[:12], (params["a"], params["b"]),
                fontsize=6, ha="center", va="bottom")
ax.scatter([a], [b], c="blue", marker="*", s=300, edgecolors="k",
           linewidth=1.5, zorder=10, label="Desmond+2023 (real)")
ax.scatter([-17.06], [-72.71], c="red", marker="D", s=100, edgecolors="k",
           zorder=10, label="SPARC")
ax.set_xlabel("CPX5 a")
ax.set_ylabel("CPX5 b")
ax.set_title("(b) CPX5 Space: Real Data vs Simulations")
ax.legend(fontsize=6, loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/desmond2023_cpx5_fit.png", dpi=150)
print(f"\nSaved {OUTDIR}/desmond2023_cpx5_fit.png")
plt.close()

# Save
np.savetxt(f"{OUTDIR}/desmond2023_digitized.csv",
           np.column_stack([x, y]),
           header="log_gbar,log_gobs", delimiter=",", comments="")
print(f"Saved {OUTDIR}/desmond2023_digitized.csv")

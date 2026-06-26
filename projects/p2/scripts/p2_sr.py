"""
P2: GW Mass Distribution — PySR Discovery.
Fit log(count) vs log(m1) using binned data with Poisson likelihood.
"""
import numpy as np
import sys
from pysr import PySRRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
print(f"Seed: {SEED}")

# Load
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'gwtc3.npz')
data = np.load(data_path)
m1 = data['events']['m1']
print(f"N = {len(m1)} events")

# Bin the data
bins_per_dex = 5
min_m = 1.0
max_m = 200.0
nbins = int(np.ceil(np.log10(max_m/min_m) * bins_per_dex))
bins = np.logspace(np.log10(min_m), np.log10(max_m), nbins + 1)
counts, _ = np.histogram(m1, bins=bins)
bin_centers = np.sqrt(bins[:-1] * bins[1:])

# Keep bins with >0 counts for log fitting
keep = counts > 0
x = np.log10(bin_centers[keep])
y = np.log10(counts[keep].astype(float))

print(f"Using {len(x)} bins (dropped {(~keep).sum()} empty)")

# Poisson error model: sqrt(N)/N for log space
yerr = np.where(counts[keep] > 0, 1.0 / np.sqrt(counts[keep]) / np.log(10), 0.1)
print(f"y range: {y.min():.1f} - {y.max():.1f}")

# === PySR ===
model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt", "square"],
    model_selection="accuracy",
    parsimony=0.01,
    populations=20,
    population_size=50,
    ncycles_per_iteration=1000,
    maxsize=15,
    random_state=SEED,
    procs=12,
    parallelism="multithreading",
    output_directory=os.path.join(script_dir, '..', 'analysis'),
    run_id=f'p2_sr_seed{SEED}',
    verbosity=1,
    progress=True,
)

print("Running PySR...")
model.fit(x.reshape(-1, 1), y, weights=1.0 / yerr**2)
print(f"Done! Best: {model.sympy()}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.errorbar(10**x, 10**y, yerr=10**y * yerr * np.log(10), fmt='o', alpha=0.5)
m_grid = np.logspace(np.log10(min_m), np.log10(max_m), 300)
xg = np.log10(m_grid)
y_pred = model.predict(xg.reshape(-1, 1))
ax.plot(m_grid, 10**y_pred, 'r-', lw=2, label='SR')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('m1 (M$_\\odot$)')
ax.set_ylabel('Count per bin')
ax.set_title(f'Mass Distribution Fit (Seed {SEED})')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
# Convert to PDF
norm = np.trapezoid(10**y_pred, m_grid)
pdf_pred = 10**y_pred / norm
count_norm = np.trapezoid(counts.astype(float), bin_centers)
pdf_data = counts / count_norm
ax.step(bin_centers, pdf_data, where='mid', alpha=0.5, label='Data')
ax.plot(m_grid, pdf_pred, 'r-', lw=2, label='SR')
ax.set_xlabel('m1 (M$_\\odot$)')
ax.set_ylabel('PDF')
ax.set_title('Probability Density')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(1, 150)

fig.tight_layout()
fig.savefig(os.path.join(script_dir, '..', 'analysis', f'sr_fit_seed{SEED}.png'), dpi=150)
print(f"Saved sr_fit_seed{SEED}.png")

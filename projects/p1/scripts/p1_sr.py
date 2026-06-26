"""
P1: Exoplanet Mass-Radius Relation — PySR Discovery.
Log-log space: log10(R/R_E) = f(log10(M/M_E))
"""
import numpy as np
from pysr import PySRRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
print(f"Seed: {SEED}")

# Load data
data = np.load('../data/exoplanets.npz')
p = data['planets']

# Filter
valid = (p['mass'] > 0.1) & (p['rad'] > 0) & ~np.isnan(p['mass']*p['rad'])
has_err = ~np.isnan(p['mass_err_low']*p['mass_err_high']*p['rad_err_low']*p['rad_err_high'])
keep = valid & has_err
print(f"Keeping {keep.sum()}/{len(p)} planets")

X = np.log10(p['mass'][keep]).reshape(-1, 1)
y = np.log10(p['rad'][keep])

# Error model: max(measurement_error, 0.1*y) for log-space intrinsic scatter
ml = np.abs(p['mass_err_low'][keep]) / p['mass'][keep] / np.log(10)  # log-space error
mh = np.abs(p['mass_err_high'][keep]) / p['mass'][keep] / np.log(10)
rl = np.abs(p['rad_err_low'][keep]) / p['rad'][keep] / np.log(10)
rh = np.abs(p['rad_err_high'][keep]) / p['rad'][keep] / np.log(10)

# Symmetrize and add intrinsic scatter floor
yerr = np.maximum((rl + rh) / 2, 0.1 * np.abs(y))
xerr = np.maximum((ml + mh) / 2, 0.01)
print(f"yerr range: {yerr.min():.4f} - {yerr.max():.4f}")

# PySR
model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt", "square"],
    model_selection="accuracy",
    parsimony=0.01,
    populations=20,
    population_size=50,
    ncycles_per_iteration=1000,
    fraction_replaced=0.01,
    fraction_replaced_hof=0.01,
    maxsize=15,
    random_state=SEED,
    procs=12,
    parallelism="multithreading",
    output_directory='../analysis/',
    run_id=f'p1_sr_seed{SEED}',
    verbosity=1,
    progress=True,
)

print("Running PySR...")
model.fit(X, y, weights=1.0 / yerr**2)
print("Done!")

# Save results
print(f"Best equation: {model.sympy()}")
print(f"Best score: {model.score(X, y)}")
print(f"Best loss: {model.loss_}")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
m_grid = np.logspace(-1, 4, 300)
ax.scatter(10**X, 10**y, c='C0', alpha=0.2, s=3, label='Data')
y_pred = model.predict(np.log10(m_grid).reshape(-1, 1))
ax.plot(m_grid, 10**y_pred, 'r-', lw=2, label='SR best')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mass (M$_\\oplus$)')
ax.set_ylabel('Radius (R$_\\oplus$)')
ax.set_title(f'SR Seed {SEED}')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'../analysis/sr_fit_seed{SEED}.png', dpi=150)
print(f"Saved analysis/sr_fit_seed{SEED}.png")

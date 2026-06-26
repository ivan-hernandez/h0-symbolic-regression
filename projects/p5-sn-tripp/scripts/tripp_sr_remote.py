#!/usr/bin/env python3
"""
SR on Hubble diagram residuals δμ = MU_SH0ES - μ(z; H0=70).
Search for non-linear f(x1, c, z) beyond linear Tripp.
Runs on remote (Julia + PySR available).
"""

import numpy as np
import pickle, os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['PATH'] = os.path.expanduser('~/julia/bin') + ':' + os.environ.get('PATH', '')
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'

from pysr import PySRRegressor

SEED = 42
CACHE = "/tmp/sn_tripp_cache"

# === Load data ===
fpath = os.path.join(CACHE, "tripp_residual.pkl")
if not os.path.exists(fpath):
    # Need to create it
    sys.path.insert(0, ".")
    import tripp_fit
    tripp_fit.main()

with open(fpath, "rb") as f:
    d = pickle.load(f)

z = d["z"]; x1 = d["x1"]; c = d["c"]
residual = d["residual"]  # cosmology-removed: only x1/c/host structure remains
rms_total = d["rms_total"]
rms_x1c = d["rms_x1c"]
n = len(z)

print(f"Data: {n} SNe")
print(f"z: [{z.min():.3f}, {z.max():.2f}]")
print(f"x1: [{x1.min():.2f}, {x1.max():.2f}]")
print(f"c: [{c.min():.3f}, {c.max():.3f}]")
print(f"RMS (total): {rms_total:.4f}")
print(f"RMS (cosmo removed): {rms_x1c:.4f} mag")

# ==========================================
# SR Run 1: residual = f(x1, c) — search for non-linear Tripp corrections
# ==========================================
print(f"\n{'='*60}")
print("Run 1: residual = f(x1, c)")
print(f"{'='*60}")

X = np.column_stack([x1, c])
model1 = PySRRegressor(
    niterations=400,
    populations=15,
    population_size=40,
    ncycles_per_iteration=300,
    maxsize=20,
    parsimony=1e-5,
    warm_start=False,
    turbo=True,
    timeout_in_seconds=72000,
    random_state=SEED,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "abs", "square", "sqrt"],
    loss="L2DistLoss()",
    model_selection="accuracy",
    output_directory="./output_tripp_x1c",
    progress=True,
    verbosity=1,
)
model1.fit(X, residual, variable_names=['x1', 'c'])

# ==========================================
# SR Run 2: residual = f(x1, c, z) — check for z-dependent Tripp
# ==========================================
print(f"\n{'='*60}")
print("Run 2: residual = f(x1, c, z)")
print(f"{'='*60}")

X2 = np.column_stack([x1, c, z])
model2 = PySRRegressor(
    niterations=400,
    populations=15,
    population_size=40,
    ncycles_per_iteration=300,
    maxsize=25,
    parsimony=1e-5,
    warm_start=False,
    turbo=True,
    timeout_in_seconds=72000,
    random_state=SEED+1,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "abs", "square", "sqrt"],
    loss="L2DistLoss()",
    model_selection="accuracy",
    output_directory="./output_tripp_x1cz",
    progress=True,
    verbosity=1,
)
model2.fit(X2, residual, variable_names=['x1', 'c', 'z'])

# ==========================================
# Save results
# ==========================================
results = {
    "n": n,
    "rms_total": float(rms_total),
    "rms_x1c": float(rms_x1c),
    "run1": [],
    "run2": [],
}
for i in range(min(10, len(model1.equations_))):
    row = model1.equations_.iloc[i]
    results["run1"].append({
        "complexity": int(row['complexity']),
        "loss": float(row['loss']),
        "equation": str(row['sympy_format']),
    })
for i in range(min(10, len(model2.equations_))):
    row = model2.equations_.iloc[i]
    results["run2"].append({
        "complexity": int(row['complexity']),
        "loss": float(row['loss']),
        "equation": str(row['sympy_format']),
    })

outfile = os.path.join(CACHE, "tripp_sr_results.pkl")
with open(outfile, "wb") as f:
    pickle.dump(results, f)

print(f"\n{'='*60}")
print("Best models — Run 1 (δμ = f(x1, c)):")
for m in results["run1"][:5]:
    print(f"  C={m['complexity']}, loss={m['loss']:.6f}: {m['equation']}")
print(f"\nBest models — Run 2 (δμ = f(x1, c, z)):")
for m in results["run2"][:5]:
    print(f"  C={m['complexity']}, loss={m['loss']:.6f}: {m['equation']}")
print(f"\nSaved to {outfile}")
print("DONE")

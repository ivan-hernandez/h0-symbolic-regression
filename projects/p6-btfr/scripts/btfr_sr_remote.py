#!/usr/bin/env python3
"""
SR discovery on the Baryonic Tully-Fisher Relation.
Search for Mbary = f(Vflat) without assuming power-law form.
"""

import numpy as np
import pickle, os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['PATH'] = os.path.expanduser('~/julia/bin') + ':' + os.environ.get('PATH', '')
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'

from pysr import PySRRegressor

SEED = 42
CACHE = "/tmp/btfr_cache"

fpath = os.path.join(CACHE, "btfr_data.pkl")
if not os.path.exists(fpath):
    sys.path.insert(0, ".")
    from btfr_fit import main as btfr_main
    btfr_main()

with open(fpath, "rb") as f:
    d = pickle.load(f)

logMb = d["logMb"]
logV = d["logV"]
Mbary = d["Mbary"]
Vflat = d["Vflat"]
n = len(logMb)

print(f"Data: {n} SPARC galaxies")
print(f"log Vflat: [{logV.min():.2f}, {logV.max():.2f}]")
print(f"log Mbary: [{logMb.min():.2f}, {logMb.max():.2f}]")
print(f"Best-fit power law: Mbary ∝ Vflat^{d['slope']:.2f}")

# ==========================================
# SR Run 1: log Mbary = f(log Vflat)
# Standard log-log space
# ==========================================
print(f"\n{'='*60}")
print("Run 1: log Mbary = f(log Vflat)")
print(f"{'='*60}")

X1 = logV.reshape(-1, 1)
model1 = PySRRegressor(
    niterations=300,
    populations=15,
    population_size=40,
    ncycles_per_iteration=300,
    maxsize=20,
    parsimony=0.001,
    warm_start=False,
    turbo=True,
    timeout_in_seconds=36000,
    random_state=SEED,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "sqrt", "cube"],
    loss="L2DistLoss()",
    model_selection="accuracy",
    output_directory="./output_btfr_log",
    progress=True,
    verbosity=1,
)
model1.fit(X1, logMb, variable_names=['logV'])

# ==========================================
# SR Run 2: Mbary = f(Vflat) — linear space
# ==========================================
print(f"\n{'='*60}")
print("Run 2: Mbary = f(Vflat) — linear space")
print(f"{'='*60}")

X2 = Vflat.reshape(-1, 1)
model2 = PySRRegressor(
    niterations=300,
    populations=15,
    population_size=40,
    ncycles_per_iteration=300,
    maxsize=25,
    parsimony=0.001,
    warm_start=False,
    turbo=True,
    timeout_in_seconds=36000,
    random_state=SEED+1,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "sqrt", "cube", "exp", "log"],
    loss="L2DistLoss()",
    model_selection="accuracy",
    output_directory="./output_btfr_lin",
    progress=True,
    verbosity=1,
)
model2.fit(X2, Mbary, variable_names=['Vflat'])

# ==========================================
# Save results
# ==========================================
results = {
    "n": n,
    "slope_linear": d["slope"],
    "rms_linear": d["rms"],
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

outfile = os.path.join(CACHE, "btfr_sr_results.pkl")
with open(outfile, "wb") as f:
    pickle.dump(results, f)

print(f"\n{'='*60}")
print("Best models — Run 1 (log-log):")
for m in results["run1"][:5]:
    print(f"  C={m['complexity']}, loss={m['loss']:.6f}: {m['equation']}")
print(f"\nBest models — Run 2 (linear):")
for m in results["run2"][:5]:
    print(f"  C={m['complexity']}, loss={m['loss']:.6f}: {m['equation']}")
print(f"\nSaved to {outfile}")
print("DONE")

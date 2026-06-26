#!/usr/bin/env python3
"""
SR discovery of NS EOS: log P = f(log nb).
Runs per-EOS and joint fits, compares with polytrope.
"""
import numpy as np
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eos_data import load_eos, load_for_sr

CACHE = "/tmp/nseos_cache"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
julia_dir = os.path.expanduser("~/julia/bin")
if os.path.exists(julia_dir):
    os.environ["PATH"] = julia_dir + ":" + os.environ.get("PATH", "")

d = load_eos()
sr = load_for_sr()

nb = sr["nb"]
logP = sr["logP"]
labels = sr["label"]
eos_names = sr["eos_names"]

# === Baseline: polytrope (log-log linear) per EOS ===
print("=== Baseline: Polytrope (log-log linear) per EOS ===")
baselines = {}
all_pred_poly = np.zeros_like(logP)
from numpy.linalg import lstsq

for name in eos_names:
    mask = labels == name
    x = np.log10(nb[mask])
    y = logP[mask]
    A = np.column_stack([np.ones_like(x), x])
    coeff, *_ = lstsq(A, y, rcond=None)
    gamma, logK = coeff[1], coeff[0]
    pred = logK + gamma * x
    rms = np.sqrt(np.mean((y - pred)**2))
    all_pred_poly[mask] = pred
    baselines[name] = {"gamma": float(gamma), "logK": float(logK), "rms": float(rms), "n": mask.sum()}
    print(f"  {name}: γ={gamma:.4f}, K=1e{logK:.3f}, RMS={rms:.4f}")

total_rms_poly = np.sqrt(np.mean((logP - all_pred_poly)**2))
print(f"  Total polytrope RMS: {total_rms_poly:.4f}")

# === SR Run 1: Per-EOS (use DDBm as representative) ===
print(f"\n=== SR Run 1: log P = f(log nb) for DDBm ===")
mask_m = labels == "DDBm"
x_m = np.log10(nb[mask_m])
y_m = logP[mask_m]
print(f"  {len(x_m)} points, nb [{x_m.min():.3f}, {x_m.max():.3f}]")
sys.stdout.flush()

from pysr import PySRRegressor

model1 = PySRRegressor(
    niterations=300,
    populations=15,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sqrt", "square", "cube", "exp", "log", "sin", "cos"],
    maxsize=25,
    complexity_of_operators={"sin": 3, "cos": 3, "exp": 4},
    constraints={"^": (-1, 1)},
    loss="L2DistLoss()",
    model_selection="accuracy",
    parsimony=0.0032,
    warm_start=False,
    batching=False,
    procs=12,
    multithreading=True,
    random_state=42,
    verbosity=1,
    progress=False,
)

model1.fit(x_m.reshape(-1, 1), y_m)

run1_results = []
for eq_idx in range(len(model1.equations_)):
    row = model1.equations_.iloc[eq_idx]
    rms = np.sqrt(float(row["loss"]))
    run1_results.append({
        "complexity": int(row["complexity"]),
        "loss": float(row["loss"]),
        "rms": float(rms),
        "equation": str(row["equation"]),
    })
    drms = rms - baselines["DDBm"]["rms"]
    drms_str = f"{drms:+.4f}" if abs(drms) > 1e-6 else "—"
    print(f"  C={int(row['complexity']):2d} RMS={rms:.4f} Δ={drms_str}  {row['equation']}")

# === SR Run 2: Joint fit across all EOSs ===
print(f"\n=== SR Run 2: log P = f(log nb) joint (all EOSs) ===")
x_all = np.log10(nb)
print(f"  {len(x_all)} points from {len(eos_names)} EOSs")
sys.stdout.flush()

model2 = PySRRegressor(
    niterations=500,
    populations=15,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sqrt", "square", "cube", "exp", "log", "sin", "cos"],
    maxsize=30,
    complexity_of_operators={"sin": 3, "cos": 3, "exp": 4},
    constraints={"^": (-1, 1)},
    loss="L2DistLoss()",
    model_selection="accuracy",
    parsimony=0.0032,
    warm_start=False,
    batching=False,
    procs=12,
    multithreading=True,
    random_state=43,
    verbosity=1,
    progress=False,
)

model2.fit(x_all.reshape(-1, 1), logP)

run2_results = []
for eq_idx in range(len(model2.equations_)):
    row = model2.equations_.iloc[eq_idx]
    rms = np.sqrt(float(row["loss"]))
    run2_results.append({
        "complexity": int(row["complexity"]),
        "loss": float(row["loss"]),
        "rms": float(rms),
        "equation": str(row["equation"]),
    })
    drms = rms - total_rms_poly
    drms_str = f"{drms:+.4f}" if abs(drms) > 1e-6 else "—"
    print(f"  C={int(row['complexity']):2d} RMS={rms:.4f} Δ={drms_str}  {row['equation']}")

# === Save ===
with open(os.path.join(CACHE, "eos_sr_results.pkl"), "wb") as f:
    pickle.dump({
        "n": len(nb),
        "eos_names": eos_names,
        "baselines": baselines,
        "total_rms_poly": float(total_rms_poly),
        "run1_ddbm": run1_results,
        "run2_joint": run2_results,
    }, f)
print(f"\nDone. Results saved to {CACHE}/eos_sr_results.pkl")

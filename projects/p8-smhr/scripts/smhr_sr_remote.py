#!/usr/bin/env python3
"""
SR discovery of baryon conversion efficiency: log ε = f(log M_h, z).
Runs SR per-z then joint 2D fit.
"""
import numpy as np
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smhr_data import load_smhr, fit_double_power_law, double_power_law

CACHE = "/tmp/smhr_cache"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
julia_dir = os.path.expanduser("~/julia/bin")
if os.path.exists(julia_dir):
    os.environ["PATH"] = julia_dir + ":" + os.environ.get("PATH", "")

d = load_smhr()
logMh = d["logMh"]
log_eps = d["log_eps"]
z_arr = d["z"]
redshifts = d["redshifts"]

# === Baseline: double power law at each z ===
print("=== Baseline Double Power Law Fits ===")
baselines = {}
all_pred_dpl = np.zeros_like(log_eps)
for zi, z in enumerate(redshifts):
    mask = np.abs(z_arr - z) < 0.01
    xz, yz = logMh[mask], log_eps[mask]
    params, rms = fit_double_power_law(xz, yz, z)
    pred = double_power_law(xz, *params)
    all_pred_dpl[mask] = pred
    baselines[z] = {"params": params.tolist(), "rms": float(rms), "n": mask.sum()}
    n_pts = baselines[z]['n']
    print(f'  z={z:.2f}: RMS={rms:.4f} ({n_pts} pts)')
    print(f"  z={z:.2f}: RMS={rms:.4f} ({n_pts} pts)\nn_pts = baselines[z]['n']")

total_rms_dpl = np.sqrt(np.mean((log_eps - all_pred_dpl)**2))
print(f"  Total DPL RMS: {total_rms_dpl:.4f}")

# === Run 1: SR at z=0.1 (fixed z, most data) ===
print(f"\n=== SR Run 1: log ε = f(log M_h) at z=0.1 ===")
mask_z0 = np.abs(z_arr - 0.10) < 0.01
x_z0, y_z0 = logMh[mask_z0], log_eps[mask_z0]
print(f"  {len(x_z0)} points, logMh [{x_z0.min():.2f}, {x_z0.max():.2f}]")
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

model1.fit(x_z0.reshape(-1, 1), y_z0)

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
    drms = rms - baselines[0.10]["rms"]
    drms_str = f"{drms:+.4f}" if abs(drms) > 1e-6 else "—"
    print(f"  C={int(row['complexity']):2d} RMS={rms:.4f} Δ={drms_str}  {row['equation']}")

# === Run 2: Joint 2D SR: log ε = f(log M_h, z) with both features ===
print(f"\n=== SR Run 2: log ε = f(log M_h, z) joint fit ===")
X_joint = np.column_stack([logMh, z_arr])
print(f"  {len(logMh)} points, {X_joint.shape[1]} features")
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

model2.fit(X_joint, log_eps)

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
    drms = rms - total_rms_dpl
    drms_str = f"{drms:+.4f}" if abs(drms) > 1e-6 else "—"
    print(f"  C={int(row['complexity']):2d} RMS={rms:.4f} Δ={drms_str}  {row['equation']}")

# === Save ===
with open(os.path.join(CACHE, "smhr_sr_results.pkl"), "wb") as f:
    pickle.dump({
        "n": len(logMh),
        "redshifts": redshifts,
        "baselines": baselines,
        "total_rms_dpl": float(total_rms_dpl),
        "run1_z0": run1_results,
        "run2_joint": run2_results,
    }, f)
print(f"\nDone. Results saved to {CACHE}/smhr_sr_results.pkl")

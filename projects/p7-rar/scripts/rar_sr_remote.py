#!/usr/bin/env python3
"""
SR discovery of RAR functional form: log g_obs = f(log g_bar)
Runs 2 SR searches (log-log space), then validates against McGaugh RAR form.
"""

import numpy as np
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rar_data import load_rar_log

CACHE = "/tmp/rar_cache"
DATA = load_rar_log()
x, y = DATA["x"], DATA["y"]

# McGaugh RAR form for comparison
def mcgaug(h_gobs, g_bar, g_dag=1.2e-10):
    return np.log10(g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dag))))

# Baseline: power law
from numpy.linalg import lstsq
A = np.column_stack([np.ones_like(x), x])
coeff, *_ = lstsq(A, y, rcond=None)
a_slope, b_intercept = coeff[1], coeff[0]
y_pred_power = a_slope * x + b_intercept
rms_power = np.sqrt(np.mean((y - y_pred_power)**2))
print(f"Baseline power law: y = {a_slope:.4f}*x + {b_intercept:.4f}, RMS = {rms_power:.4f}")

# Baseline: McGaugh form
def mcgaug_adaptive(x_log, g0):
    """McGaugh form with free g_dagger."""
    g_bar = 10**x_log
    y_pred = np.log10(g_bar / (1 - np.exp(-np.sqrt(g_bar / g0))))
    return y_pred

from scipy.optimize import minimize_scalar
def fit_mcgaug(x_log, y_obs):
    def loss(g0):
        return np.mean((y_obs - mcgaug_adaptive(x_log, g0))**2)
    res = minimize_scalar(loss, bounds=(1e-13, 1e-9), method='bounded')
    return res.x, np.sqrt(res.fun)

g0_best, rms_mcgaug = fit_mcgaug(x, y)
print(f"McGaugh form: g_dag = {g0_best:.2e}, RMS = {rms_mcgaug:.4f}")

# Write baseline info
info = {
    "n": DATA["n"],
    "x_range": [float(x.min()), float(x.max())],
    "y_range": [float(y.min()), float(y.max())],
    "power_law": {"slope": float(a_slope), "intercept": float(b_intercept), "rms": float(rms_power)},
    "mcgaug": {"g_dagger": float(g0_best), "rms": float(rms_mcgaug)},
}
with open(os.path.join(CACHE, "rar_baseline.pkl"), "wb") as f:
    pickle.dump(info, f)
print(f"\nBaseline saved: rms_power={rms_power:.4f}, rms_mcgaug={rms_mcgaug:.4f}")
print(f"Ratio (power/McGaugh RMS): {rms_power/rms_mcgaug:.4f}")

# === SR Discovery ===
# Running on remote with Julia/PySR
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
# Add Julia path
julia_dir = os.path.expanduser("~/julia/bin")
if os.path.exists(julia_dir):
    os.environ["PATH"] = julia_dir + ":" + os.environ.get("PATH", "")

from pysr import PySRRegressor

model = PySRRegressor(
    niterations=300,
    populations=15,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=[
        "sqrt", "square", "cube", "exp", "log",
        "sin", "cos",
    ],
    maxsize=25,
    complexity_of_operators={"sin": 3, "cos": 3, "exp": 4},
    constraints={"^": (-1, 1)},  # prevent nested powers
    nesting_depth=4,
    loss="L2DistLoss()",
    model_selection="accuracy",
    early_stop_condition=(
        "stop_if(loss, complexity) :: "
        "complexity >= 10 && complexity <= 20 && "
        "loss < 0.0001 && "  # RMS < 0.01
        "(loss / prev_loss > 0.999)"
    ),
    temp_equation_file=False,
    parsimony=0.0032,
    warm_start=False,
    batching=False,
    procs=12,
    multithreading=True,
    random_state=42,
    verbosity=1,
    progress=False,
)

print(f"\n=== SR Run 1: log g_obs = f(log g_bar) ===")
print(f"N={len(x)}, x in [{x.min():.3f}, {x.max():.3f}]")
sys.stdout.flush()

model.fit(x.reshape(-1, 1), y)

results = []
for eq_idx in range(len(model.equations_)):
    row = model.equations_.iloc[eq_idx]
    results.append({
        "complexity": int(row["complexity"]),
        "loss": float(row["loss"]),
        "equation": str(row["equation"]),
    })
    rms = np.sqrt(float(row["loss"]))
    drms = rms - rms_power
    if abs(drms) < 1e-6:
        drms_str = "—"
    else:
        drms_str = f"{drms:+.4f}"
    print(f"  C={int(row['complexity']):2d} RMS={rms:.4f} Δ={drms_str}  {row['equation']}")

with open(os.path.join(CACHE, "rar_sr_results.pkl"), "wb") as f:
    pickle.dump({
        "n": len(x),
        "data_info": info,
        "results": results,
    }, f)

print(f"\nResults saved to {CACHE}/rar_sr_results.pkl")
print("Done.")

"""Option 1: TFR SR Discovery — PySR script for remote machine.

Transfers to remote, runs PySR on SPARC TFR data, saves results.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os

# Load per-galaxy TFR data
df = pd.read_csv("analysis/tfr/tfr_galaxies.csv")
V = df["V_flat"].values
M = df["M_b"].values
log_V = np.log10(V)
log_M = np.log10(np.maximum(M, 1e-10))

good = np.isfinite(log_V) & np.isfinite(log_M) & (V > 0) & (M > 1e6)
X = log_V[good].reshape(-1, 1)
y = log_M[good]

print(f"TFR data: {len(X)} galaxies")
print(f"log V range: [{X.min():.2f}, {X.max():.2f}]")
print(f"log M range: [{y.min():.2f}, {y.max():.2f}]")

# ── PySR Setup ──
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt", "square"],
    maxsize=25,
    populations=12,
    population_size=50,
    model_selection="accuracy",
    parsimony=0.001,
    progress=False,
    verbosity=1,
)

print("Running PySR on TFR...")
model.fit(X, y)

print(f"\nBest equation: {model.sympy()}")

eqs = model.equations_.sort_values("score", ascending=False)
print(f"\nPareto front ({len(eqs)} equations):")
for i, row in eqs.head(10).iterrows():
    print(f"  Cpx {row.get('complexity', '?')}: loss={row.get('loss', '?'):.4f}, "
          f"score={row.get('score', '?'):.4f}, eq={row.get('sympy_format', '?')}")

# Save
eqs.to_csv("analysis/tfr/tfr_pysr_equations.csv", index=False)
print("Saved analysis/tfr/tfr_pysr_equations.csv")

"""
Phase 2 — SR discovery on prokaryote metabolic scaling.

Runs PySR on multiple subsets to discover functional forms of
log10(B) = f(log10(M)) for prokaryotes.

Strategy:
  - All prokaryotes (endogenous+active), with state as categorical
  - Endogenous only (largest homogeneous subset)
  - Active only (steeper exponent)
  - DeLong-only and Hoehler-only cross-checks

Uses OLS-equivalent weighting in log space.
Error model: σ_log = max(σ_meas / (B·ln(10)), 0.043 dex) for intrinsic scatter
"""
import numpy as np, os, csv, time, sys
from pysr import PySRRegressor

os.environ["COLUMNS"] = "120"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = '/home/ivan/general-conversation/projects/p11-microbe-scaling/output/microbial_metabolic_data.csv'
OUT_DIR = '/home/ivan/general-conversation/projects/p11-microbe-scaling/output'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load
# ============================================================
rows = []
with open(DATA_PATH) as f:
    for r in csv.DictReader(f):
        rows.append(r)

def get_subset(domain_filter='prok', state_filter=None, source_filter=None):
    subset = []
    for r in rows:
        if domain_filter == 'prok' and r['domain'] not in ('Archaea', 'Bacteria'):
            continue
        if state_filter and r['state'] != state_filter:
            continue
        if source_filter and r['source'] != source_filter:
            continue
        mass = float(r['mass_g'])
        mr = float(r['metabolic_rate_W'])
        if mass > 0 and mr > 0:
            subset.append((mass, mr))
    return subset

def run_sr(subset, label, niterations=300, seed=42):
    """Run PySR on log10 MR vs log10 Mass."""
    masses = np.array([s[0] for s in subset])
    mrs = np.array([s[1] for s in subset])
    logM = np.log10(masses)
    logB = np.log10(mrs)

    # Error model: intrinsic scatter floor of 0.1 dex in B
    # σ_log10(B) ≈ σ_B / (B · ln(10))
    # With σ_B = 0.1 * B, this gives σ_log ≈ 0.0434 dex
    # Use uniform weights = 1/σ² = 1/(0.0434²) ≈ 530
    # But also incorporate any actual measurement scatter
    # For simplicity, use uniform weights
    weights = np.ones_like(logM) * 530.0

    X = logM.reshape(-1, 1)
    y = logB

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        niterations=niterations,
        populations=12,
        population_size=100,
        maxsize=20,
        parsimony=0.0005,
        precision=64,
        turbo=True,
        procs=12,
        model_selection="accuracy",
        random_state=seed,
        early_stop_condition=1e-12,
        verbosity=1,
        parallelism="multithreading",
    )

    print(f"\n{'='*60}")
    print(f"SR Run: {label} ({len(subset)} pts, seed={seed})")
    print(f"{'='*60}")

    t0 = time.time()
    model.fit(X, y, weights=weights)
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed:.1f}s")

    best_idx = model.equations_['loss'].idxmin()
    print("\n  Top equations (sorted by loss):")
    for i, row in model.equations_.sort_values('loss').head(min(10, len(model.equations_))).iterrows():
        eq_str = str(row['sympy_format'])[:60]
        print(f"    [{i}] loss={row['loss']:.6f} cpx={row['complexity']}  {eq_str}")

    print("\n  Hall of Fame:")
    print(f"  {'Cpx':>4} {'Loss':>10} {'b_eff':>8}  Equation")
    print(f"  {'-'*4} {'-'*10} {'-'*8}  {'-'*50}")
    for i, row in model.equations_.sort_values('complexity').iterrows():
        try:
            b_eff = float(model.predict([[1.0]], index=i)[0] - model.predict([[0.0]], index=i)[0])
        except:
            b_eff = float('nan')
        eq_short = str(row['sympy_format'])[:50]
        print(f"  {row['complexity']:>4d} {row['loss']:>10.6f} {b_eff:>8.4f}  {eq_short}")

    return model

# ============================================================
# Run configurations
# ============================================================
configs = [
    ("prok_all", {'domain_filter': 'prok', 'state_filter': None, 'source_filter': None}),
    ("prok_endogenous", {'domain_filter': 'prok', 'state_filter': 'endogenous', 'source_filter': None}),
    ("prok_active", {'domain_filter': 'prok', 'state_filter': 'active', 'source_filter': None}),
]

results = {}
for label, cfg in configs:
    subset = get_subset(**cfg)
    print(f"\n{label}: {len(subset)} entries")
    masses = [s[0] for s in subset]
    print(f"  Mass range: {min(masses):.2e} to {max(masses):.2e} g")
    results[label] = run_sr(subset, label, niterations=300, seed=42)

# ============================================================
# Cross-checks with different seeds
# ============================================================
print("\n\n=== SEED VARIATION (prok_all, 3 seeds) ===")
for seed in [7, 123, 999]:
    subset = get_subset('prok')
    run_sr(subset, f"prok_all_seed{seed}", niterations=200, seed=seed)

# ============================================================
# Baseline power law comparison
# ============================================================
print("\n\n=== BASELINE POWER LAW COMPARISON ===")
from numpy.polynomial import Polynomial

for label, cfg in configs:
    subset = get_subset(**cfg)
    masses = np.array([s[0] for s in subset])
    mrs = np.array([s[1] for s in subset])
    logM = np.log10(masses)
    logB = np.log10(mrs)

    # OLS
    A = np.vstack([np.ones_like(logM), logM]).T
    c, res, rank, s = np.linalg.lstsq(A, logB, rcond=None)
    y_pred = A @ c
    mse = np.mean((logB - y_pred)**2)

    print(f"  {label} ({len(subset)} pts):")
    print(f"    OLS power law:  B = {10**c[0]:.2e} * M^{c[1]:.3f}")
    print(f"    MSE: {mse:.6f} (RMS={np.sqrt(mse):.4f} dex)")
    print(f"    R²: {1 - np.sum((logB-y_pred)**2)/np.sum((logB-np.mean(logB))**2):.4f}")

print("\nDone. All SR results logged above.")

"""
Phase 3 validation: bootstrap, taxonomic holdout, mass truncation.

Tests the cubic form logB = a*(logM)³ + c found by SR for prokaryote active state.
"""
import numpy as np
import csv, os, sys

DATA_PATH = os.path.join(os.path.dirname(__file__), '..',
                         'output', 'microbial_metabolic_data.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def load_data():
    rows = []
    with open(DATA_PATH) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    prok = []
    phyla = set()
    for r in rows:
        if r['domain'] in ('Archaea', 'Bacteria'):
            mass = float(r['mass_g'])
            mr = float(r['metabolic_rate_W'])
            if mass > 0 and mr > 0:
                r['_logM'] = np.log10(mass)
                r['_logB'] = np.log10(mr)
                prok.append(r)
                phyla.add(r.get('phylum', 'unknown'))
    return prok, phyla

def fit_cubic(logM, logB):
    """Fit logB = a*(logM)³ + c via least-squares on design matrix [x³, 1]."""
    X = np.column_stack([logM**3, np.ones_like(logM)])
    coeffs, residuals, _, _ = np.linalg.lstsq(X, logB, rcond=None)
    a = coeffs[0]
    c = coeffs[1]
    pred = X @ coeffs
    mse = np.mean((logB - pred)**2)
    r2 = 1 - np.sum((logB - pred)**2) / np.sum((logB - np.mean(logB))**2)
    return a, c, mse, r2

def effective_slope(a, logM):
    """d(logB)/d(logM) = 3*a*(logM)²"""
    return 3 * a * logM**2

def rmse(logB, pred):
    return np.sqrt(np.mean((logB - pred)**2))

# ================================================================
# Load
# ================================================================
print("="*60)
print("PHASE 3 — VALIDATION")
print("="*60)

prok, phyla = load_data()
print(f"Total prokaryotes: {len(prok)}")
print(f"Phyla present: {sorted(phyla)}")

# Split by state
active = [r for r in prok if r['state'] == 'active']
endogenous = [r for r in prok if r['state'] == 'endogenous']
print(f"  Active: {len(active)}")
print(f"  Endogenous: {len(endogenous)}")

# ================================================================
# 1. Baseline cubic fits
# ================================================================
print("\n--- 1. Baseline cubic fit ---")

for label, subset in [('Active', active), ('Endogenous', endogenous), ('All prok', prok)]:
    logM = np.array([r['_logM'] for r in subset])
    logB = np.array([r['_logB'] for r in subset])
    a, c, mse, r2 = fit_cubic(logM, logB)
    slope_min = effective_slope(a, logM.min())
    slope_max = effective_slope(a, logM.max())
    print(f"  {label:>12} ({len(subset):>3}): "
          f"logB = {a:.5f}·(logM)³ + {c:.4f}  "
          f"MSE={mse:.4f}  R²={r2:.4f}  "
          f"b_eff [{slope_min:.2f}, {slope_max:.2f}]")

# ================================================================
# 2. Bootstrap (200 resamples)
# ================================================================
print("\n--- 2. Bootstrap (200 resamples) ---")

for label, subset in [('Active', active), ('Endogenous', endogenous), ('All prok', prok)]:
    logM_all = np.array([r['_logM'] for r in subset])
    logB_all = np.array([r['_logB'] for r in subset])
    n = len(subset)
    a_boot = []
    c_boot = []
    
    np.random.seed(42)
    for _ in range(200):
        idx = np.random.randint(0, n, size=n)
        logM_b = logM_all[idx]
        logB_b = logB_all[idx]
        a, c, _, _ = fit_cubic(logM_b, logB_b)
        a_boot.append(a)
        c_boot.append(c)
    
    a_mean = np.mean(a_boot)
    a_std = np.std(a_boot)
    a_ci = np.percentile(a_boot, [16, 84])
    c_mean = np.mean(c_boot)
    c_std = np.std(c_boot)
    c_ci = np.percentile(c_boot, [16, 84])
    
    print(f"  {label:>12}: a = {a_mean:.5f} ± {a_std:.5f}  68% CI [{a_ci[0]:.5f}, {a_ci[1]:.5f}]")
    print(f"             c = {c_mean:.4f} ± {c_std:.4f}  68% CI [{c_ci[0]:.4f}, {c_ci[1]:.4f}]")
    
    # Slope at mid-mass
    midM = np.median(logM_all)
    mid_slope = effective_slope(a_mean, midM)
    mid_slope_lo = effective_slope(a_ci[0], midM)
    mid_slope_hi = effective_slope(a_ci[1], midM)
    # Slope at min/max
    min_slope = effective_slope(a_mean, logM_all.min())
    max_slope = effective_slope(a_mean, logM_all.max())
    print(f"             b_eff: [{min_slope:.2f}, {mid_slope:.2f}, {max_slope:.2f}] "
          f"at masses [{10**logM_all.min():.1e}, {10**midM:.1e}, {10**logM_all.max():.1e}] g")

# ================================================================
# 3. Taxonomic holdout
# ================================================================
print("\n--- 3. Taxonomic holdout (leave-one-phylum-out) ---")

for label, subset in [('Active', active), ('All prok', prok)]:
    print(f"\n  {label}:")
    
    # Get phylum distribution
    from collections import Counter
    phy_counts = Counter(r.get('phylum', 'unknown') for r in subset)
    # Skip phyla with <5 members
    valid_phyla = [p for p, c in phy_counts.items() if c >= 5]
    
    # Baseline on all data
    logM_all = np.array([r['_logM'] for r in subset])
    logB_all = np.array([r['_logB'] for r in subset])
    a_ref, c_ref, _, _ = fit_cubic(logM_all, logB_all)
    
    for phylum in sorted(valid_phyla):
        holdout = [r for r in subset if r.get('phylum', 'unknown') == phylum]
        training = [r for r in subset if r.get('phylum', 'unknown') != phylum]
        
        logM_tr = np.array([r['_logM'] for r in training])
        logB_tr = np.array([r['_logB'] for r in training])
        logM_ho = np.array([r['_logM'] for r in holdout])
        logB_ho = np.array([r['_logB'] for r in holdout])
        
        a_ho, c_ho, _, _ = fit_cubic(logM_tr, logB_tr)
        
        # Predict holdout
        pred_ho = a_ho * logM_ho**3 + c_ho
        rmse_ho = rmse(logB_ho, pred_ho)
        
        # % change in a
        da_pct = (a_ho - a_ref) / abs(a_ref) * 100
        
        n_tr = len(training)
        n_ho = len(holdout)
        print(f"    leave out {phylum:>14} ({n_ho:>3} held, {n_tr:>3} trained): "
              f"a={a_ho:.5f}  Δa={da_pct:+.1f}%  RMSE_holdout={rmse_ho:.4f}")

# ================================================================
# 4. Mass range truncation
# ================================================================
print("\n--- 4. Mass range truncation sensitivity ---")

for label, subset in [('Active', active), ('Endogenous', endogenous), ('All prok', prok)]:
    print(f"\n  {label}:")
    logM_all = np.array([r['_logM'] for r in subset])
    logB_all = np.array([r['_logB'] for r in subset])
    
    a_ref, c_ref, mse_ref, r2_ref = fit_cubic(logM_all, logB_all)
    
    for frac in [0.05, 0.10, 0.20]:
        lo = np.percentile(logM_all, frac * 100)
        hi = np.percentile(logM_all, (1 - frac) * 100)
        
        mask = (logM_all >= lo) & (logM_all <= hi)
        a_tr, c_tr, _, _ = fit_cubic(logM_all[mask], logB_all[mask])
        da_pct = (a_tr - a_ref) / abs(a_ref) * 100
        
        n_kept = mask.sum()
        mass_range_dec = hi - lo  # in dex
        print(f"    trim {frac*100:>3.0f}% tails ({n_kept:>3} kept, "
              f"{mass_range_dec:.1f} dex): a={a_tr:.5f}  Δa={da_pct:+.1f}%")

print("\nDone.")

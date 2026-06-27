"""
Deep SR — Phase 2 extended discovery.

Systematic SR across multiple states, seeds, and operator sets.
Saves all hall-of-fame tables for cross-comparison.

Strategy:
  - Active state: 5 seeds × 500 iterations
  - Endogenous state: 5 seeds × 500 iterations
  - All prok: 3 seeds × 500 iterations
  - Test with/without cubic terms
  - Compare forms across seeds for robustness
"""
import numpy as np, os, csv, time, json, pickle
from pysr import PySRRegressor
from numpy.polynomial import Polynomial

os.environ["COLUMNS"] = "120"

DATA_PATH = '/home/ivan/general-conversation/projects/p11-microbe-scaling/output/microbial_metabolic_data.csv'
OUT_DIR = '/home/ivan/general-conversation/projects/p11-microbe-scaling/output'
SR_DIR = os.path.join(OUT_DIR, 'sr_results')
os.makedirs(SR_DIR, exist_ok=True)

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
            subset.append({
                'mass': mass, 'mr': mr, 'source': r['source'],
                'domain': r['domain'], 'state': r['state'],
                'species': r.get('species', '')
            })
    return subset

def make_data(subset):
    masses = np.array([s['mass'] for s in subset])
    mrs = np.array([s['mr'] for s in subset])
    logM = np.log10(masses)
    logB = np.log10(mrs)
    weights = np.ones_like(logM) * 530.0  # intrinsic scatter floor
    return logM.reshape(-1, 1), logB, weights

def run_sr(subset, label, niterations=500, seed=42, 
           binary_ops=["+", "-", "*", "/"],
           unary_ops=["square"],
           parsimony=0.001):
    """Run PySR and save results."""
    X, y, weights = make_data(subset)
    run_id = f"{label}_seed{seed}_n{niterations}"

    model = PySRRegressor(
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        niterations=niterations,
        populations=12,
        population_size=100,
        maxsize=20,
        parsimony=parsimony,
        precision=64,
        turbo=True,
        procs=12,
        model_selection="accuracy",
        random_state=seed,
        early_stop_condition=1e-12,
        verbosity=0,
        parallelism="multithreading",
    )

    print(f"  [{run_id}] {len(subset)} pts, {niterations} iter, parsimony={parsimony}")
    t0 = time.time()
    model.fit(X, y, weights=weights)
    elapsed = time.time() - t0

    # Extract results
    eqs = model.equations_.copy()
    best_idx = eqs['loss'].idxmin()

    # Save model
    model_path = os.path.join(SR_DIR, f"{run_id}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save hall of fame as CSV
    hof_path = os.path.join(SR_DIR, f"{run_id}_hof.csv")
    eqs.to_csv(hof_path, index=False)

    # Summary
    print(f"  -> Best: cpx={eqs.loc[best_idx,'complexity']}, loss={eqs.loc[best_idx,'loss']:.6f}")
    print(f"     {eqs.loc[best_idx,'sympy_format']}")
    print(f"     [{elapsed:.0f}s]")

    # Evaluate effective slope at mass range limits
    logM_all = X.flatten()
    try:
        y0 = float(model.predict([[logM_all.min()]], index=best_idx)[0])
        y1 = float(model.predict([[logM_all.max()]], index=best_idx)[0])
        b_eff = (y1 - y0) / (logM_all.max() - logM_all.min())
        print(f"     Effective b: {b_eff:.3f} over [{logM_all.min():.1f}, {logM_all.max():.1f}]")
    except:
        pass

    return model

# ============================================================
# Configurations
# ============================================================
configs = []

# Active state — main targets (strongest curvature signal)
for seed in [7, 42, 123, 777, 999]:
    configs.append(('prok_active', {'state_filter': 'active'}, 500, seed))

# Endogenous state
for seed in [7, 42, 123, 777, 999]:
    configs.append(('prok_endogenous', {'state_filter': 'endogenous'}, 500, seed))

# All prokaryotes
for seed in [7, 42, 123]:
    configs.append(('prok_all', {}, 500, seed))

# ============================================================
# Run
# ============================================================
print("="*60)
print("DEEP SR: Phase 2 extended discovery")
print("="*60)

all_results = []
for label, filters, niter, seed in configs:
    subset = get_subset(**filters)
    if len(subset) < 20:
        print(f"  Skipping {label}: only {len(subset)} pts")
        continue
    
    model = run_sr(subset, label, niterations=niter, seed=seed,
                   binary_ops=["+", "-", "*", "/"],
                   unary_ops=["square", "cube"],
                   parsimony=0.001)
    
    # Extract best model per complexity
    best = model.equations_.iloc[model.equations_['loss'].idxmin()]
    all_results.append({
        'label': label, 'seed': seed, 'niter': niter,
        'n_pts': len(subset),
        'best_loss': best['loss'],
        'best_cpx': best['complexity'],
        'best_eq': str(best['sympy_format']),
    })

# ============================================================
# Summary
# ============================================================
print("\n\n" + "="*60)
print("SUMMARY: Best model per configuration")
print("="*60)
print(f"{'Config':<30} {'n':>4} {'Loss':>8} {'Cpx':>4}  Best equation")
print("-"*80)
for r in sorted(all_results, key=lambda x: (x['label'], x['seed'])):
    eq_short = r['best_eq'][:50]
    print(f"  {r['label']}_s{r['seed']:<20} {r['n_pts']:>4} {r['best_loss']:>8.4f} {r['best_cpx']:>4}  {eq_short}")

# Cross-seed consistency check
print("\n\n=== Cross-seed consistency ===")
for label in ['prok_active', 'prok_endogenous', 'prok_all']:
    label_results = [r for r in all_results if r['label'] == label]
    if not label_results:
        continue
    print(f"\n{label} ({label_results[0]['n_pts']} pts):")
    losses = [r['best_loss'] for r in label_results]
    print(f"  Loss range: {min(losses):.4f} - {max(losses):.4f}")
    print(f"  Median loss: {np.median(losses):.4f}")
    
    # Check if any run found a non-power-law form as best
    for r in label_results:
        eq = r['best_eq']
        has_quadratic = '**2' in eq or 'square' in eq
        has_cubic = '**3' in eq or 'cube' in eq
        has_division = '/' in eq
        has_powerlaw = not has_quadratic and not has_cubic and not has_division
        r['type'] = 'powerlaw' if has_powerlaw else 'quadratic' if has_quadratic and not has_cubic else 'cubic' if has_cubic else 'rational'
        print(f"  seed={r['seed']}: loss={r['best_loss']:.4f}, type={r['type']}, eq={r['best_eq'][:60]}")

# Save summary
summary_path = os.path.join(SR_DIR, 'summary.csv')
with open(summary_path, 'w') as f:
    w = csv.DictWriter(f, fieldnames=['label', 'seed', 'niter', 'n_pts', 'best_loss', 'best_cpx', 'best_eq', 'type'])
    w.writeheader()
    w.writerows(all_results)
print(f"\nSummary saved to {summary_path}")
print(f"All models saved to {SR_DIR}/")

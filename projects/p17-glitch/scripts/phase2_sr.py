"""Phase 2: SR discover optimal form of glitch size distribution.

Target: CCDF (complementary cumulative distribution) ~ f(log10(size))
Predictors: log10(Δν/ν)
"""
import os, sys, math
import numpy as np
from scipy import stats

DATAFILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'glitch_sizes.txt')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTDIR, exist_ok=True)

# Load
sizes = np.loadtxt(DATAFILE, comments='#')
log_sizes = np.log10(sizes)
n = len(sizes)

# CCDF
sorted_idx = np.argsort(sizes)[::-1]
sorted_log = np.log10(sizes[sorted_idx])
ccdf = np.arange(1, n + 1) / n

# Fit power law as baseline
res = stats.linregress(sorted_log, np.log10(ccdf))
alpha = 1 - res.slope
print(f'=== Glitch Size Distribution ===')
print(f'Glitches: {n}')
print(f'Power law fit: α = {alpha:.3f}, R² = {res.rvalue**2:.4f}')

# Prepare for SR: X = log10(size), y = log10(CCDF)
X = sorted_log.reshape(-1, 1)
y = np.log10(ccdf)

# Run SR
try:
    import pysr
except ImportError:
    print('\nPySR not available.')
    sys.exit(1)

for seed in [42, 123, 777]:
    print(f'\n=== SR seed {seed} ===')
    model = pysr.PySRRegressor(
        niterations=500,
        populations=20,
        model_selection='accuracy',
        maxsize=12,
        parsimony=0.001,
        batching=False,
        warm_start=False,
        verbosity=0,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['inv', 'neg', 'exp', 'square', 'cube'],
        random_state=seed,
    )
    model.fit(X, y)
    print(model)
    model.equations_.to_csv(os.path.join(OUTDIR, f'sr_seed{seed}_hof.csv'), index=False)

print('\nDone. Check output/ for results.')

"""Phase 2: SR discover optimal SNIa correction form.

Target: Δμ(residual after z-removal) ~ f(x1, c, host_mass)
"""
import csv, os, math, sys
import numpy as np

DATAFILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'pantheon_plus.csv')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTDIR, exist_ok=True)

C = 299792.458

def mu_lcdm(z, H0=68.0, Om=0.321):
    Ol = 1 - Om
    n = 200
    h = z / (2*n)
    xs = [i*h for i in range(2*n+1)]
    fx = [1/math.sqrt(Om*(1+x)**3+Ol) for x in xs]
    integral = h/3 * (fx[0] + fx[-1] + 4*sum(fx[1::2]) + 2*sum(fx[2:-1:2]))
    Dc = C / 100 * integral
    DL = (1+z) * Dc
    return 5*math.log10(max(DL, 1e-10)) + 25

# Load and compute residuals
rows = []
with open(DATAFILE) as f:
    for row in csv.DictReader(f):
        z = float(row['zHD'])
        mu = float(row['MU_SH0ES'])
        x1 = float(row['x1'])
        c = float(row['c'])
        host = float(row['HOST_LOGMASS']) if row['HOST_LOGMASS'] and row['HOST_LOGMASS'] != '-9' else float('nan')
        if z > 0.01:
            rows.append({'z': z, 'mu': mu, 'x1': x1, 'c': c, 'host': host})

mu_mod = np.array([mu_lcdm(r['z']) for r in rows])
mu_dat = np.array([r['mu'] for r in rows])
res = mu_dat - mu_mod

z = np.array([r['z'] for r in rows])
x1 = np.array([r['x1'] for r in rows])
c = np.array([r['c'] for r in rows])
h = np.array([r['host'] if not math.isnan(r['host']) else 0 for r in rows])
has_host = np.array([not math.isnan(r['host']) for r in rows])

# Remove cosmology (z-dependent) residual
A_z = np.column_stack([z, z**2, np.ones_like(z)])
c_z = np.linalg.lstsq(A_z, res, rcond=None)[0]
res_target = res - A_z @ c_z

# Prepare SR features: x1, c, host (use 0 for missing host)
feature_names = ['x1', 'c', 'host']
X = np.column_stack([
    x1,               # stretch
    c,                # color
    h,                # host mass (0 if missing)
])
y = res_target

# Filter to valid entries
mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
X, y = X[mask], y[mask]

print('Data: %d SNe, features: %s' % (len(y), feature_names))
print('y range: [%.4f, %.4f]' % (np.min(y), np.max(y)))

# Run PySR with x1, c, host as predictors
try:
    import pysr
except ImportError:
    print('\nPySR not available.')
    sys.exit(1)

seeds = [42, 123, 777]
for seed in seeds:
    print('\n=== SR seed %d ===' % seed)
    model = pysr.PySRRegressor(
        niterations=500,
        populations=20,
        model_selection='accuracy',
        maxsize=15,
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
    model.equations_.to_csv(os.path.join(OUTDIR, 'sr_seed%d_hof.csv' % seed), index=False)

print('\nDone. Check output/ for results.')

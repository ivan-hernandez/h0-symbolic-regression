"""Phase 2: SR discover optimal dark matter halo profile.

Data: v_dm(r) = sqrt(v_obs^2 - v_bar^2) from SPARC.
Strategy: normalize by R_max, V_max, discover universal form.
Compare to NFW, Einasto, Burkert.
"""
import csv, os, math, glob
import numpy as np
from scipy import stats, optimize

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTDIR, exist_ok=True)

def read_galaxy(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                r = float(parts[0])
                vobs = float(parts[1])
                vgas = float(parts[2])
                vdisk = float(parts[3])
                vbul = float(parts[4])
            except ValueError:
                continue
            if vobs > 0:
                rows.append({'r': r, 'vobs': vobs, 'vgas': vgas, 'vdisk': vdisk, 'vbul': vbul})
    return rows

# Load and compute v_dm
galaxies = []
for f in sorted(glob.glob(os.path.join(DATADIR, '*_rotmod.dat'))):
    data = read_galaxy(f)
    if len(data) < 5:
        continue
    name = os.path.basename(f).replace('_rotmod.dat', '')
    r = np.array([d['r'] for d in data])
    vobs = np.array([d['vobs'] for d in data])
    vbar = np.sqrt(np.array([d['vgas']**2 + d['vdisk']**2 + d['vbul']**2 for d in data]))
    vdm2 = vobs**2 - vbar**2
    vdm = np.sqrt(np.maximum(vdm2, 0))
    galaxies.append({'name': name, 'r': r, 'vobs': vobs, 'vbar': vbar, 'vdm': vdm})

# Normalize by galaxy
all_x, all_y = [], []
for g in galaxies:
    r = g['r']
    vdm = g['vdm']
    r_max = np.max(r)
    v_max = np.max(vdm) if np.max(vdm) > 0 else 1
    if v_max <= 0:
        continue
    x = r / r_max
    y = vdm / v_max
    all_x.extend(x.tolist())
    all_y.extend(y.tolist())

X = np.array(all_x).reshape(-1, 1)
y = np.array(all_y)

print('Data: %d points from %d galaxies' % (len(y), len(galaxies)))
print('X (r/Rmax) range: [%.3f, %.3f]' % (np.min(X), np.max(X)))
print('y (vdm/Vmax) range: [%.3f, %.3f]' % (np.min(y), np.max(y)))

# Baseline fits
xf = X.flatten()
print('\n=== Baseline fits ===')

# Linear
res = stats.linregress(xf, y)
y_pred_l = res.slope * xf + res.intercept
mse_l = np.mean((y - y_pred_l)**2)
print('Linear: y = %.4f*x %+.4f, R=%.4f, MSE=%.6f' % (res.slope, res.intercept, res.rvalue**2, mse_l))

# Quadratic
A2 = np.column_stack([xf**2, xf, np.ones_like(xf)])
c2 = np.linalg.lstsq(A2, y, rcond=None)[0]
y_pred_q = A2 @ c2
mse_q = np.mean((y - y_pred_q)**2)
print('Quadratic: MSE=%.6f' % mse_q)

# Power law in x: y = a * x^b
try:
    popt, _ = optimize.curve_fit(lambda x, a, b: a * x**b, xf, y, p0=[1, 0.5])
    y_pred_pow = popt[0] * xf**popt[1]
    mse_pow = np.mean((y - y_pred_pow)**2)
    print('Power law: y = %.4f * x^%.4f, MSE=%.6f' % (popt[0], popt[1], mse_pow))
except:
    mse_pow = 1e10

# AIC comparison
n = len(y)
for name, mse, k in [('Linear', mse_l, 2), ('Quadratic', mse_q, 3), ('Power law', mse_pow, 3)]:
    if mse > 0:
        aic = n * math.log(mse) + 2 * k
        print('  %-12s AIC=%.1f' % (name, aic))

# Run PySR
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
        binary_operators=['+', '-', '*', '/', '^'],
        unary_operators=['inv', 'neg', 'exp', 'square', 'cube', 'sqrt'],
        random_state=seed,
    )
    model.fit(X, y)
    print(model)
    model.equations_.to_csv(os.path.join(OUTDIR, 'sr_seed%d_hof.csv' % seed), index=False)

print('\n=== Result ===')
print('If SR recovers power law or NFW-like form -> consistent with literature')
print('If SR discovers novel form -> potentially NOVEL')
print('Check results in output/ directory')

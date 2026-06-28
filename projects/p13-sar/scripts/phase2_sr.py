"""Phase 2: SR discovery of the optimal SAR functional form.

Strategy: demean by study to remove intercept offsets, then
SR discovers f(logA) for logS in log-log space.

Multiple seeds, 500 iterations, model_selection='accuracy'.
Compares: linear (power law) vs quadratic vs rational vs exp forms.
"""
import csv, os, sys, math, json
import numpy as np
from collections import defaultdict
from scipy import stats

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTDIR, exist_ok=True)

def load_and_prepare():
    # Load island data, forward-fill study
    studies = defaultdict(list)
    with open(os.path.join(DATADIR, 'SAR_island.csv'), encoding='latin-1') as f:
        current_study = ''
        for row in csv.DictReader(f):
            if row.get('STUDYREP', '').strip():
                current_study = row['STUDYREP'].strip()
            try:
                s = float(row.get('S', 0))
                a = float(row.get('Akm2', 0))
            except:
                continue
            if a > 0 and s > 0:
                studies[current_study].append({'area': a, 'species': s})

    # Only studies with >=5 points
    good_studies = {k: v for k, v in studies.items() if len(v) >= 5}

    # Demean by study
    all_x, all_y = [], []
    for study, pts in good_studies.items():
        logA = np.array([math.log10(p['area']) for p in pts])
        logS = np.array([math.log10(p['species']) for p in pts])
        mean_logA = np.mean(logA)
        mean_logS = np.mean(logS)
        for la, ls in zip(logA, logS):
            all_x.append(la - mean_logA)
            all_y.append(ls - mean_logS)

    X = np.array(all_x).reshape(-1, 1)
    y = np.array(all_y)
    return X, y, len(good_studies)

def baseline_fits(X, y):
    xf = X.flatten()
    print('\n=== Baseline fits (pooled, demeaned) ===')

    # Linear (power law)
    res = stats.linregress(xf, y)
    y_pred = res.slope * xf + res.intercept
    mse_l = np.mean((y - y_pred)**2)
    print('Linear (power law): y = %.4f*x %+.4f, R=%.4f, MSE=%.6f' % (res.slope, res.intercept, res.rvalue**2, mse_l))

    # Quadratic
    A2 = np.column_stack([xf**2, xf, np.ones_like(xf)])
    coeff2 = np.linalg.lstsq(A2, y, rcond=None)[0]
    y_pred_q = A2 @ coeff2
    mse_q = np.mean((y - y_pred_q)**2)
    print('Quadratic:       y = %.6f*x^2 %+.6f*x %+.6f, MSE=%.6f' % (coeff2[0], coeff2[1], coeff2[2], mse_q))

    # Cubic
    A3 = np.column_stack([xf**3, xf**2, xf, np.ones_like(xf)])
    coeff3 = np.linalg.lstsq(A3, y, rcond=None)[0]
    y_pred_c = A3 @ coeff3
    mse_c = np.mean((y - y_pred_c)**2)
    print('Cubic:           MSE=%.6f' % mse_c)

    # Fourth order
    A4 = np.column_stack([xf**4, xf**3, xf**2, xf, np.ones_like(xf)])
    coeff4 = np.linalg.lstsq(A4, y, rcond=None)[0]
    y_pred_4 = A4 @ coeff4
    mse_4 = np.mean((y - y_pred_4)**2)
    print('4th order:       MSE=%.6f' % mse_4)

    # AIC
    n = len(y)
    def aic(mse, k):
        return n * math.log(mse) if mse > 0 else -n * math.log(n)
    if mse_l > 0 and mse_q > 0 and mse_c > 0 and mse_4 > 0:
        for name, mse, k in [('Linear', mse_l, 2), ('Quadratic', mse_q, 3),
                              ('Cubic', mse_c, 4), ('4th order', mse_4, 5)]:
            print('  %-12s AIC=%.1f' % (name, n * math.log(mse) + 2*k))

    improvement = (mse_l - mse_q) / mse_l * 100
    print('\nQuadratic improvement over linear: %.1f%%' % improvement)
    return mse_l, mse_q

def main():
    X, y, n_studies = load_and_prepare()
    print('Data: %d points from %d studies' % (len(y), n_studies))
    print('X range: [%.3f, %.3f]' % (np.min(X), np.max(X)))
    print('y range: [%.3f, %.3f]' % (np.min(y), np.max(y)))

    mse_l, mse_q = baseline_fits(X, y)

    # Run PySR
    try:
        import pysr
    except ImportError:
        print('\nPySR not available.')
        return

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

        hof = model.equations_
        hof.to_csv(os.path.join(OUTDIR, 'sr_seed%d_hof.csv' % seed), index=False)

    print('\n=== Final result ===')
    print('If SR finds linear (power law) is best -> KILLED')
    print('If SR finds quadratic/cubic/inv -> NOVEL (proceed to Phase 3)')

if __name__ == '__main__':
    main()

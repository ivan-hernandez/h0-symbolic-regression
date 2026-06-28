"""Phase 2: SR discovery of the temperature-rate functional form.

Strategy: demean by species to remove intercept offsets, then
SR discovers f(inv_kT) for log10(rate) in Arrhenius space.

Multiple seeds, 500 iterations, model_selection='accuracy'.
"""
import csv, os, sys, math, json
import numpy as np

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def load_and_prepare():
    meta = {}
    with open(os.path.join(DATADIR, 'meta_data.csv')) as f:
        for row in csv.DictReader(f):
            ds = (row.get('Data set') or '').strip()
            meta[ds] = row

    records = []
    with open(os.path.join(DATADIR, 'metabolic_data.csv')) as f:
        for row in csv.DictReader(f):
            temp_str = (row.get('Temp (C)') or '').strip()
            rate_w_str = (row.get('WO Metabolic rate (W)') or '').strip()
            sp = (row.get('Scientific name') or '').strip()
            ds = (row.get('Data set') or '').strip()
            if not all([temp_str, rate_w_str, sp]):
                continue
            try:
                temp_c = float(temp_str)
                rate_w = float(rate_w_str)
            except (ValueError, TypeError):
                continue
            if rate_w <= 0:
                continue
            T_k = temp_c + 273.15
            inv_kT = 1.0 / (8.617333262e-5 * T_k)
            log_rate = math.log10(rate_w)
            records.append({
                'species': sp,
                'temp_c': temp_c,
                'inv_kT': inv_kT,
                'log_rate': log_rate,
                'mass': None,
            })

    if not records:
        sys.exit('No data loaded')

    # Demean by species
    sp_means = {}
    for r in records:
        sp = r['species']
        if sp not in sp_means:
            sp_means[sp] = {'log_rate': [], 'inv_kT': []}
        sp_means[sp]['log_rate'].append(r['log_rate'])
        sp_means[sp]['inv_kT'].append(r['inv_kT'])

    sp_mean_y = {}
    sp_mean_x = {}
    for sp, vals in sp_means.items():
        sp_mean_y[sp] = np.mean(vals['log_rate'])
        sp_mean_x[sp] = np.mean(vals['inv_kT'])

    X, y = [], []
    for r in records:
        X.append(r['inv_kT'] - sp_mean_x[r['species']])
        y.append(r['log_rate'] - sp_mean_y[r['species']])

    return np.array(X).reshape(-1, 1), np.array(y)

def main():
    X, y = load_and_prepare()
    print('Data: %d points' % len(y))
    print('X shape:', X.shape)
    print('y range: [%.3f, %.3f]' % (np.min(y), np.max(y)))
    print('X range: [%.3f, %.3f]' % (np.min(X), np.max(X)))

    # Try fitting PySR
    try:
        import pysr
    except ImportError:
        print('\nPySR not installed. Running simple baseline fits instead.')
        return baseline_fits(X, y)

    seeds = [42, 123, 777, 999, 7]
    for seed in seeds[:3]:
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

        # Save HOF
        hof = model.equations_
        hof.to_csv(os.path.join(OUTDIR, 'sr_seed%d_hof.csv' % seed), index=False)

def baseline_fits(X, y):
    y_flat = y.flatten() if hasattr(y, 'flatten') else X  # placeholder
    print('\n=== Baseline fits ===')

    # Linear
    from scipy import stats
    xf = X.flatten()
    res = stats.linregress(xf, y)
    y_pred = res.slope * xf + res.intercept
    mse_lin = np.mean((y - y_pred)**2)
    print('Linear: y = %.4f*x %+.4f, R=%.4f, MSE=%.6f' % (res.slope, res.intercept, res.rvalue**2, mse_lin))

    # Quadratic
    A = np.column_stack([xf**2, xf, np.ones_like(xf)])
    coeff = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred_q = A @ coeff
    mse_quad = np.mean((y - y_pred_q)**2)
    print('Quadratic: y = %.6f*x^2 %+.6f*x %+.6f, MSE=%.6f' % (coeff[0], coeff[1], coeff[2], mse_quad))

    # Cubic
    A3 = np.column_stack([xf**3, xf**2, xf, np.ones_like(xf)])
    coeff3 = np.linalg.lstsq(A3, y, rcond=None)[0]
    y_pred_c = A3 @ coeff3
    mse_cubic = np.mean((y - y_pred_c)**2)
    print('Cubic:    MSE=%.6f' % mse_cubic)

    # AIC comparison
    n = len(y)
    def aic(mse, k):
        return n * math.log(mse) + 2 * k

    aic_lin = aic(mse_lin, 2)
    aic_quad = aic(mse_quad, 3)
    aic_cubic = aic(mse_cubic, 4)
    print('\nAIC comparison:')
    print('  Linear:   %.1f' % aic_lin)
    print('  Quadratic: %.1f (vs linear: %.1f)' % (aic_quad, aic_lin - aic_quad))
    print('  Cubic:    %.1f (vs linear: %.1f)' % (aic_cubic, aic_lin - aic_cubic))

    # Best model
    best = min([('Linear', aic_lin), ('Quadratic', aic_quad), ('Cubic', aic_cubic)], key=lambda x: x[1])
    print('\nBest model by AIC: %s' % best[0])

    if best[0] == 'Linear':
        print('Result: No curvature detected. MTE linear Arrhenius is optimal.')
    else:
        print('Result: Curvature detected! Non-linear form preferred.')

if __name__ == '__main__':
    main()

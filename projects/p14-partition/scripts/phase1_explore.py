"""Phase 1: Explore p(n) vs Hardy-Ramanujan asymptotic."""
import csv, os, math
import numpy as np
from scipy import stats

DATAFILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'p_n.csv')

# Load
ns, log10p, sqrts = [], [], []
with open(DATAFILE) as f:
    for row in csv.DictReader(f):
        n = int(row['n'])
        if n < 1:
            continue
        ns.append(n)
        log10p.append(float(row['log10_p_n']))
        sqrts.append(math.sqrt(n))

log10p = np.array(log10p)
sqrts = np.array(sqrts)
ns = np.array(ns)

print('=== Phase 1: Partition Function p(n) ===')
print('n in [%d, %d]' % (ns[0], ns[-1]))
print('log10 p(n) in [%.3f, %.3f]' % (log10p[0], log10p[-1]))

# Hardy-Ramanujan: p(n) ~ exp(pi*sqrt(2n/3)) / (4*n*sqrt(3))
# log10 p(n) = log10(exp(pi*sqrt(2n/3))) - log10(4*n*sqrt(3))
#           = pi*sqrt(2/3) * sqrt(n) / ln(10) - log10(n) - log10(4*sqrt(3))
hr_pi = math.pi * math.sqrt(2/3) / math.log(10)
hr_log_n_coeff = -1.0  # coefficient of log10(n)
hr_const = -math.log10(4 * math.sqrt(3))

log10p_hr = hr_pi * sqrts + hr_log_n_coeff * np.log10(ns) + hr_const
residuals = log10p - log10p_hr

print('\n=== Hardy-Ramanujan comparison ===')
print('HR slope (pi*sqrt(2/3)/ln(10)): %.6f' % hr_pi)
print('HR log(n) coefficient: %.1f' % hr_log_n_coeff)
print('HR constant: %.6f' % hr_const)
print('Mean residual: %.6f' % np.mean(residuals))
print('Std residual: %.6f' % np.std(residuals))
print('Max residual: %.6f' % np.max(np.abs(residuals)))

# Fit log10 p(n) vs sqrt(n) (without the log n term)
res = stats.linregress(sqrts, log10p)
print('\n=== Naive linear fit: log10 p(n) = a*sqrt(n) + b ===')
print('a = %.6f (HR predicts %.6f)' % (res.slope, hr_pi))
print('b = %.6f' % res.intercept)
print('R = %.8f' % res.rvalue**2)

# Fit with sqrt(n) and log10(n)
A = np.column_stack([sqrts, np.log10(ns), np.ones_like(ns)])
coeff = np.linalg.lstsq(A, log10p, rcond=None)[0]
print('\n=== Fit: log10 p = a*sqrt(n) + b*log10(n) + c ===')
print('a = %.6f (HR: %.6f)' % (coeff[0], hr_pi))
print('b = %.6f (HR: %.1f)' % (coeff[1], hr_log_n_coeff))
print('c = %.6f (HR: %.6f)' % (coeff[2], hr_const))
print('R = %.8f' % (1 - np.var(log10p - A @ coeff) / np.var(log10p)))

pred = A @ coeff
resid2 = log10p - pred
print('Residual std: %.6f' % np.std(resid2))

# Check if the residuals have structure (i.e., missing term)
print('\n=== Residual analysis ===')
# AIC for HR vs best-fit
n = len(ns)
mse_hr = np.mean(residuals**2)
mse_fit = np.mean(resid2**2)
aic_hr = n * math.log(mse_hr) + 6
aic_fit = n * math.log(mse_fit) + 8
print('AIC HR model: %.1f' % aic_hr)
print('AIC best-fit: %.1f' % aic_fit)
print('dAIC: %.1f (HR - best)' % (aic_hr - aic_fit))
print('HR captured %.2f%% of variance' % ((1 - mse_hr/np.var(log10p))*100))

# What's the optimal coefficient? (n from 100 to NMAX to avoid small-n effects)
for n_cut in [100, 1000, 10000, 100000]:
    mask = ns >= n_cut
    if mask.sum() < 10:
        continue
    x = sqrts[mask]
    y = log10p[mask]
    n_sub = ns[mask]
    A_sub = np.column_stack([x, np.log10(n_sub), np.ones_like(x)])
    c_sub = np.linalg.lstsq(A_sub, y, rcond=None)[0]
    print(f'\nOptimal fit for n >= {n_cut}:')
    print(f'  a = {c_sub[0]:.6f} (HR: {hr_pi:.6f})')
    print(f'  b = {c_sub[1]:.6f} (HR: {hr_log_n_coeff:.1f})')
    print(f'  c = {c_sub[2]:.6f} (HR: {hr_const:.6f})')

"""Proper validation of glitch size distribution."""
import os, math
import numpy as np
from scipy import stats, optimize

DATAFILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'glitch_sizes.txt')
sizes = np.loadtxt(DATAFILE, comments='#')
n = len(sizes)

print(f'N = {n} glitches')

# MLE helpers
def w_nll(params, d):
    k, lam = params
    if k <= 0 or lam <= 0:
        return 1e10
    return -np.sum(stats.weibull_min.logpdf(d, k, scale=lam))

def fit_weibull(data, guess=(0.5, 1e-6)):
    r = optimize.minimize(lambda p: w_nll(p, data), guess, method='Nelder-Mead',
                         options={'maxiter':10000, 'xatol':1e-8, 'fatol':1e-8})
    return r.x, r.fun

# 1. Weibull MLE
print('\n1. Weibull MLE')
(k_ml, lam_ml), nll_w = fit_weibull(sizes)
print(f'   k = {k_ml:.4f}, λ = {lam_ml:.2e}, NLL = {nll_w:.1f}')

# Bootstrap
k_b, l_b = [], []
for _ in range(500):
    idx = np.random.randint(0, n, n)
    try:
        (k, lam), _ = fit_weibull(sizes[idx], (k_ml, lam_ml))
        k_b.append(k); l_b.append(lam)
    except: pass
print(f'   Bootstrap k = {np.mean(k_b):.4f} +/- {np.std(k_b):.4f}')
print(f'   Bootstrap λ = {np.mean(l_b):.2e} +/- {np.std(l_b):.2e}')

# 2. Power law (Pareto) MLE
x_min = np.min(sizes)
alpha = 1 + n / np.sum(np.log(sizes / x_min))
nll_pl = -np.sum(stats.pareto.logpdf(sizes, alpha, scale=x_min))
print(f'\n2. Power law: α = {alpha:.4f}, NLL = {nll_pl:.1f}')

# 3. Lognormal
s_ln, _, scale_ln = stats.lognorm.fit(sizes, floc=0)
nll_ln = -np.sum(stats.lognorm.logpdf(sizes, s_ln, 0, scale_ln))
print(f'\n3. Lognormal: s = {s_ln:.4f}, scale = {scale_ln:.2e}, NLL = {nll_ln:.1f}')

# 4. Exponential
lam_exp = np.mean(sizes)
nll_exp = -np.sum(stats.expon.logpdf(sizes, scale=lam_exp))
print(f'\n4. Exponential: λ = {lam_exp:.2e}, NLL = {nll_exp:.1f}')

# 5. Correct AIC
print(f'\n5. AIC comparison:')
models = [('Weibull', nll_w, 2), ('Power law', nll_pl, 1),
          ('Lognormal', nll_ln, 2), ('Exponential', nll_exp, 1)]
best = min(models, key=lambda m: 2*m[1] + 2*m[2])
for name, nll, k in models:
    aic = 2*nll + 2*k
    daic = aic - (2*best[1] + 2*best[2])
    print(f'   {name:15s} AIC = {aic:8.1f}  ΔAIC = {daic:+8.1f}')

# 6. Truncation stability
print(f'\n6. Truncation stability:')
for thr in [1e-9, 1e-8, 1e-7]:
    sub = sizes[sizes >= thr]
    if len(sub) < 10: continue
    (k, lam), _ = fit_weibull(sub, (k_ml, lam_ml))
    print(f'   ≥{thr:.0e}: n={len(sub):3d}, k={k:.3f}, λ={lam:.2e}')

# 7. Jackknife
print(f'\n7. Jackknife (remove largest):')
for rem in [0, 1, 3, 5, 10]:
    sub = np.sort(sizes)[::-1][rem:]
    (k, lam), _ = fit_weibull(sub, (k_ml, lam_ml))
    print(f'   remove top {rem:2d}: k={k:.3f}, λ={lam:.2e}')

# 8. KS tests
print(f'\n8. KS tests:')
for name, dist, args in [('Weibull', 'weibull_min', (k_ml, 0, lam_ml)),
                          ('Pareto', 'pareto', (alpha, 0, x_min)),
                          ('Exponential', 'expon', (0, lam_exp))]:
    ks, p = stats.kstest(sizes, dist, args=args)
    print(f'   {name:15s} KS={ks:.4f} p={p:.4f}')

print(f'\n---')
print(f'Best: {best[0]}')
print(f'Weibull k = {np.mean(k_b):.4f} +/- {np.std(k_b):.4f}')
print(f'Weibull λ = {np.mean(l_b):.2e} +/- {np.std(l_b):.2e}')

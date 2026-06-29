"""Phase 1: Explore pulsar glitch size distribution.

Tests: power law, lognormal, broken power law, exponential.
"""
import csv, os, math
import numpy as np
from scipy import stats

DATAFILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'glitch_sizes.txt')

# Load
sizes = np.loadtxt(DATAFILE, comments='#')
log_sizes = np.log10(sizes)

print('=== Pulsar Glitch Size Distribution ===')
print(f'Total glitches: {len(sizes)}')
print(f'Size range: [{np.min(sizes):.2e}, {np.max(sizes):.2e}]')
print(f'log10 range: [{np.min(log_sizes):.2f}, {np.max(log_sizes):.2f}]')
print(f'Mean log10(Δν/ν): {np.mean(log_sizes):.2f}')
print(f'Std log10(Δν/ν): {np.std(log_sizes):.2f}')

# 1. Power law test: P(>x) ∝ x^(-α+1)
# Sort by size
sorted_sizes = np.sort(sizes)[::-1]
n = len(sorted_sizes)
# Complementary CDF: P(>x) = rank / n
ranks = np.arange(1, n + 1)
ccdf = ranks / n

# Fit power law to tail: log(CCDF) = C - alpha * log(x)
# Use only top 50% to avoid small-number noise in tail
tail_frac = 0.5
n_tail = int(n * tail_frac)
log_x = np.log10(sorted_sizes[:n_tail])
log_ccdf = np.log10(ccdf[:n_tail])
res = stats.linregress(log_x, log_ccdf)
print(f'\n=== Power law fit (top {tail_frac*100:.0f}%) ===')
print(f'α = {-res.slope:.3f} +/- {res.stderr:.3f}')
print(f'log10(C) = {res.intercept:.3f}')
print(f'R² = {res.rvalue**2:.4f}')

# 2. Lognormal test: log10(Δν) ~ N(μ, σ)
print(f'\n=== Lognormal fit ===')
mu = np.mean(log_sizes)
sigma = np.std(log_sizes)
print(f'μ = {mu:.3f}, σ = {sigma:.3f}')

# KS test for lognormal
ks_stat, ks_p = stats.kstest(log_sizes, 'norm', args=(mu, sigma))
print(f'KS statistic: {ks_stat:.4f}, p = {ks_p:.4f}')

# 3. Broken power law
print(f'\n=== Broken power law ===')
# Find break point by splitting at each possible point and minimizing total chi2
best_break = None
best_chi2 = 1e10
for i in range(10, n - 10):
    x1 = log_sizes[:i]
    y1 = log_ccdf[:i]  # approximate
    # Actually let me use CCDF properly
    x_low = log_sizes[n-i:]  
    pass

# Simpler: test if residuals from power law show structure
residuals = log_ccdf - (res.slope * log_x + res.intercept)
rho, p = stats.spearmanr(log_x, residuals)
print(f'Residual correlation with log(x): ρ = {rho:.3f}, p = {p:.4f}')
if p < 0.05:
    print('  -> Residuals have structure: power law is rejected')
else:
    print('  -> Residuals are random: power law fits well')

# 4. AIC comparison
print(f'\n=== Model comparison ===')
# Power law (LL at data points)
log_lik_power = np.sum(np.log(stats.pareto.pdf(sizes, -res.slope + 1, scale=np.min(sizes))))
aic_power = -2 * log_lik_power + 4
# Lognormal
log_lik_logn = np.sum(np.log(stats.lognorm.pdf(sizes, sigma, scale=10**mu)))
aic_logn = -2 * log_lik_logn + 4
print(f'AIC power law: {aic_power:.1f}')
print(f'AIC lognormal: {aic_logn:.1f}')

# 5. Histogram
print(f'\n=== Histogram (log10 bins) ===')
hist, edges = np.histogram(log_sizes, bins='auto')
for i in range(len(hist)):
    if hist[i] > 0:
        print(f'  [{edges[i]:.1f}, {edges[i+1]:.1f}]: {hist[i]}')

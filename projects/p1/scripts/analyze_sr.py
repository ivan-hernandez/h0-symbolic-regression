"""
Analyze SR results: compare power law vs broken power law vs SR forms.
"""
import numpy as np
from scipy import optimize as opt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('../data/exoplanets.npz')
p = data['planets']

valid = (p['mass'] > 0.1) & (p['rad'] > 0)
has_err = ~np.isnan(p['mass_err_low']*p['mass_err_high']*p['rad_err_low']*p['rad_err_high'])
keep = valid & has_err
print(f"Using {keep.sum()} planets")

x = np.log10(p['mass'][keep])
y = np.log10(p['rad'][keep])

# Error model
yh = np.abs(p['rad_err_high'][keep]); yl = np.abs(p['rad_err_low'][keep])
yerr = np.maximum((yh + yl) / 2 / (p['rad'][keep] * np.log(10)), 0.1 * np.abs(y))

def aic(chisq, k, n):
    return chisq + 2*k

# Models
models = {}

# 1: Single power law: R = A * M^b
def pl(m, b, a):
    return b * m + a

# Fit
popt, pcov = opt.curve_fit(pl, x, y, sigma=yerr, absolute_sigma=True, p0=[0.33, 0.0])
resid = y - pl(x, *popt)
chisq = np.sum((resid / yerr)**2)
n_params = len(popt)
models['Power law (C3)'] = {'params': popt, 'chisq': chisq, 'aic': aic(chisq, n_params, len(x)), 'k': n_params}
print(f"Power law: R = M^{popt[0]:.4f} * 10^({popt[1]:.4f})")
print(f"  χ² = {chisq:.1f}, dof = {len(x)-n_params}, AIC = {aic(chisq, n_params, len(x)):.1f}")

# 2: Broken power law (2 segments)
def bpl(m, b1, b2, m_break, a):
    return np.where(m < m_break, b1 * (m - m_break) + a, b2 * (m - m_break) + a)

popt2, pcov2 = opt.curve_fit(bpl, x, y, sigma=yerr, absolute_sigma=True,
                              p0=[0.3, 0.5, np.log10(20), 0.0],
                              bounds=([0.0, 0.0, -2, -1], [1.0, 1.0, 3, 1]))
resid2 = y - bpl(x, *popt2)
chisq2 = np.sum((resid2 / yerr)**2)
n_params2 = len(popt2)
models['Broken PL'] = {'params': popt2, 'chisq': chisq2, 'aic': aic(chisq2, n_params2, len(x)), 'k': n_params2}
print(f"\nBroken PL: R = M^{popt2[0]:.4f} (M < {10**popt2[2]:.1f} M_E), M^{popt2[1]:.4f} (M > {10**popt2[2]:.1f} M_E)")
print(f"  χ² = {chisq2:.1f}, AIC = {aic(chisq2, n_params2, len(x)):.1f}")

# 3: Triple power law (rocky, transition, gas giant)
def tpl(m, b1, b2, b3, m12, m23, a):
    """b1: low, b2: mid, b3: high, break at m12, m23"""
    return np.piecewise(m, [m < m12, (m >= m12) & (m < m23), m >= m23],
                         [lambda m: b1*(m-m12)+a, lambda m: b2*(m-m12)+a,
                          lambda m: b3*(m-m23) + b2*(m23-m12) + a])

try:
    popt3, _ = opt.curve_fit(tpl, x, y, sigma=yerr, absolute_sigma=True,
                              p0=[0.2, 0.5, 0.8, np.log10(5), np.log10(100), 0.0],
                              maxfev=10000)
    resid3 = y - tpl(x, *popt3)
    chisq3 = np.sum((resid3 / yerr)**2)
    models['Triple PL'] = {'params': popt3, 'chisq': chisq3, 'aic': aic(chisq3, len(popt3), len(x)), 'k': len(popt3)}
    print(f"\nTriple PL: χ² = {chisq3:.1f}, AIC = {aic(chisq3, len(popt3), len(x)):.1f}")
except Exception as e:
    print(f"\nTriple PL fit failed: {e}")

# 4: Quadratic in log-log (smooth transition)
def quad(m, a, b, c):
    return a + b*m + c*m**2

popt4, _ = opt.curve_fit(quad, x, y, sigma=yerr, absolute_sigma=True, p0=[0.0, 0.3, 0.0])
resid4 = y - quad(x, *popt4)
chisq4 = np.sum((resid4 / yerr)**2)
models['Quadratic (C6)'] = {'params': popt4, 'chisq': chisq4, 'aic': aic(chisq4, len(popt4), len(x)), 'k': len(popt4)}
print(f"\nQuad (smooth): a={popt4[0]:.4f}, b={popt4[1]:.4f}, c={popt4[2]:.4f}")
print(f"  χ² = {chisq4:.1f}, AIC = {aic(chisq4, len(popt4), len(x)):.1f}")

# 5: Seed 42 C9 form (with pole removed): y = x/2.6455 + a
def c9_simple(m, a):
    return m / 2.6455 + a

popt5, _ = opt.curve_fit(c9_simple, x, y, sigma=yerr, absolute_sigma=True, p0=[0.0])
resid5 = y - c9_simple(x, *popt5)
chisq5 = np.sum((resid5 / yerr)**2)
models['C9 (seed42, no pole)'] = {'params': popt5, 'chisq': chisq5, 'aic': aic(chisq5, 1, len(x)), 'k': 1}
print(f"\nC9 simple (x/2.6455+a): χ² = {chisq5:.1f}, AIC = {aic(chisq5, 1, len(x)):.1f}")

# 6: Local slope evolution (running slope estimate)
# Sort by mass and compute local slope in sliding windows
order = np.argsort(x)
xs = x[order]; ys = y[order]; ye = yerr[order]
window = max(100, len(x) // 20)
slopes = np.zeros(len(x) - window)
centers = np.zeros(len(x) - window)
for i in range(len(x) - window):
    xw = xs[i:i+window]; yw = ys[i:i+window]; yew = ye[i:i+window]
    A = np.vstack([xw, np.ones_like(xw)]).T
    W = np.diag(1/yew**2)
    try:
        beta = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ yw
        slopes[i] = beta[0]
        centers[i] = np.median(xw)
    except:
        pass

print("\n\nSummary:")
best_aic = min(m['aic'] for m in models.values())
print(f"{'Model':25s} {'Params':>6s} {'AIC':>10s} {'ΔAIC':>8s}")
for name, m in sorted(models.items(), key=lambda x: x[1]['aic']):
    daic = m['aic'] - best_aic
    print(f"{name:25s} {m['k']:6d} {m['aic']:10.1f} {daic:>+8.1f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: data + best fits
m_grid = np.logspace(-1, 4, 300)
x_grid = np.log10(m_grid)
ax1.scatter(10**x, 10**y, c='C0', alpha=0.15, s=2)
ax1.plot(m_grid, 10**pl(x_grid, *models['Power law (C3)']['params']), 'r-', lw=2, label='Power law')
if 'Broken PL' in models:
    ax1.plot(m_grid, 10**bpl(x_grid, *models['Broken PL']['params']), 'g--', lw=2, label='Broken PL')
if 'Quadratic (C6)' in models:
    ax1.plot(m_grid, 10**quad(x_grid, *models['Quadratic (C6)']['params']), 'm:', lw=2, label='Quadratic')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('Mass (M$_\\oplus$)'); ax1.set_ylabel('Radius (R$_\\oplus$)')
ax1.set_title('M-R Relation: Model Comparison')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# Right: local slope vs mass
ax2.plot(10**centers, slopes, 'k-', lw=2, label='Running slope')
ax2.axhline(models['Power law (C3)']['params'][0], color='r', ls='--', label=f'PL={models["Power law (C3)"]["params"][0]:.3f}')
ax2.set_xscale('log')
ax2.set_xlabel('Mass (M$_\\oplus$)')
ax2.set_ylabel('Local power-law slope')
ax2.set_title('Slope Evolution with Mass')
ax2.axhline(0.28, color='gray', ls=':', alpha=0.5, label='Rocky (Rogers 2015)')
ax2.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Gas giant')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('../analysis/model_comparison.png', dpi=150)
print("\nSaved analysis/model_comparison.png")

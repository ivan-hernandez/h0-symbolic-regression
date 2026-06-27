"""
Generate P11 paper PDF for OSF Preprints submission.
"""
import numpy as np
import csv, os, textwrap, io, base64
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

np.random.seed(42)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..',
                         'output', 'microbial_metabolic_data.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Data loading ----
def load():
    rows = []
    with open(DATA_PATH) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    prok = []
    for r in rows:
        if r['domain'] in ('Archaea', 'Bacteria'):
            mass = float(r['mass_g'])
            mr = float(r['metabolic_rate_W'])
            if mass > 0 and mr > 0:
                r['_logM'] = np.log10(mass)
                r['_logB'] = np.log10(mr)
                prok.append(r)
    return prok

prok = load()
active = [r for r in prok if r['state'] == 'active']
endog = [r for r in prok if r['state'] == 'endogenous']

def fit_cubic(logM, logB):
    X = np.column_stack([logM**3, np.ones_like(logM)])
    coeffs = np.linalg.lstsq(X, logB, rcond=None)[0]
    pred = X @ coeffs
    mse = np.mean((logB - pred)**2)
    r2 = 1 - np.sum((logB - pred)**2) / np.sum((logB - np.mean(logB))**2)
    return coeffs[0], coeffs[1], mse, r2, pred

def fit_linear(logM, logB):
    X = np.column_stack([logM, np.ones_like(logM)])
    coeffs = np.linalg.lstsq(X, logB, rcond=None)[0]
    pred = X @ coeffs
    mse = np.mean((logB - pred)**2)
    return coeffs[0], coeffs[1], mse, pred

# ---- Figure 1: Main scatter plot with cubic fits ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (label, subset, color) in zip(axes, [
    ('Active', active, '#e74c3c'),
    ('Endogenous', endog, '#3498db'),
]):
    logM = np.array([r['_logM'] for r in subset])
    logB = np.array([r['_logB'] for r in subset])
    sources = [r['source'] for r in subset]
    
    # Plot points with source markers
    for src, marker, alpha in [('Hoehler+2023', 'o', 0.7), ('DeLong+2010', 's', 0.7)]:
        mask = np.array([s == src for s in sources])
        ax.scatter(logM[mask], logB[mask], c=color, marker=marker, 
                   alpha=alpha, s=20, label=src, edgecolors='none')
    
    # Cubic fit
    a, c, mse, r2, pred = fit_cubic(logM, logB)
    x_smooth = np.linspace(logM.min(), logM.max(), 200)
    y_smooth = a * x_smooth**3 + c
    ax.plot(x_smooth, y_smooth, 'k-', lw=2, label=f'Cubic fit (a={a:.4f})')
    
    # Linear fit for comparison
    b_lin, c_lin, mse_lin, pred_lin = fit_linear(logM, logB)
    y_lin = b_lin * x_smooth + c_lin
    ax.plot(x_smooth, y_lin, 'k--', lw=1, alpha=0.5, label=f'Linear (b={b_lin:.2f})')
    
    ax.set_xlabel('log₁₀(Mass [g])', fontsize=11)
    ax.set_ylabel('log₁₀(Metabolic Rate [W])', fontsize=11)
    ax.set_title(f'{label} (n={len(subset)})', fontsize=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.text(0.05, 0.95, f'Cubic MSE={mse:.4f}', transform=ax.transAxes, 
            fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.05, 0.85, f'Linear MSE={mse_lin:.4f}', transform=ax.transAxes,
            fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_scatter.png'), dpi=200)
plt.close()

# ---- Figure 2: Effective slope as function of mass ----
fig, ax = plt.subplots(figsize=(8, 5))

for label, subset, color in [('Active', active, '#e74c3c'), ('Endogenous', endog, '#3498db')]:
    logM = np.array([r['_logM'] for r in subset])
    logB = np.array([r['_logB'] for r in subset])
    a, c, _, _, _ = fit_cubic(logM, logB)
    
    # Effective slope = d(logB)/d(logM) = 3a(logM)²
    x = np.linspace(logM.min(), logM.max(), 200)
    slope = 3 * a * x**2
    
    mass = 10**x
    ax.plot(mass, slope, '-', c=color, lw=2, label=f'{label} (a={a:.4f})')
    
    # Fill between logM range
    ax.fill_between(mass, slope, alpha=0.1, color=color)

ax.axhline(1.0, color='gray', ls=':', alpha=0.5, label='Isometry (b=1)')
ax.axhline(0.75, color='gray', ls='--', alpha=0.3, label='Kleiber (b=3/4)')
ax.set_xscale('log')
ax.set_xlabel('Mass [g]', fontsize=11)
ax.set_ylabel('Effective log-log slope b(M)', fontsize=11)
ax.set_title('Effective Scaling Exponent vs Body Mass', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(5e-15, 2e-7)
ax.set_ylim(0, 2.5)
ax.grid(True, alpha=0.3)

# Add annotation for slope interpretation
ax.annotate('Genome-driven\nsuperlinear', xy=(1e-14, 2.1), fontsize=8, ha='left',
            xytext=(2e-14, 2.3), arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
ax.annotate('Surface-area\nlimited', xy=(5e-8, 0.5), fontsize=8, ha='left',
            xytext=(1e-8, 0.3), arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_slope.png'), dpi=200)
plt.close()

# ---- Figure 3: Bootstrap distribution ----
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (label, subset) in zip(axes, [
    ('Active', active), ('Endogenous', endog), ('All prokaryotes', prok)
]):
    logM = np.array([r['_logM'] for r in subset])
    logB = np.array([r['_logB'] for r in subset])
    n = len(subset)
    a_boot = []
    for _ in range(2000):
        idx = np.random.randint(0, n, size=n)
        a, _, _, _, _ = fit_cubic(logM[idx], logB[idx])
        a_boot.append(a)
    a_boot = np.array(a_boot)
    
    ax.hist(a_boot, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(a_boot), color='red', ls='-', lw=2, label=f'Mean={np.mean(a_boot):.5f}')
    ax.axvline(np.percentile(a_boot, 16), color='red', ls='--', lw=1)
    ax.axvline(np.percentile(a_boot, 84), color='red', ls='--', lw=1)
    ax.set_xlabel('Cubic coefficient a', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(f'{label}\na = {np.mean(a_boot):.5f} ± {np.std(a_boot):.5f}', fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_bootstrap.png'), dpi=200)
plt.close()

# ---- Figure 4: Mass truncation sensitivity ----
fig, ax = plt.subplots(figsize=(7, 4.5))

for label, subset, color, marker in [
    ('Active', active, '#e74c3c', 'o'),
    ('Endogenous', endog, '#3498db', 's'),
    ('All prok', prok, '#2ecc71', '^')
]:
    logM = np.array([r['_logM'] for r in subset])
    logB = np.array([r['_logB'] for r in subset])
    a_ref, _, _, _, _ = fit_cubic(logM, logB)
    
    fracs = np.linspace(0, 0.25, 10)
    da = []
    for frac in fracs:
        lo = np.percentile(logM, frac * 100)
        hi = np.percentile(logM, (1 - frac) * 100)
        mask = (logM >= lo) & (logM <= hi)
        a_tr, _, _, _, _ = fit_cubic(logM[mask], logB[mask])
        da.append((a_tr - a_ref) / abs(a_ref) * 100)
    
    ax.plot(fracs * 100, da, f'-{marker}', c=color, lw=1.5, label=label, markersize=4)

ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Fraction of data trimmed from each tail [%]', fontsize=10)
ax.set_ylabel('Δa [%]', fontsize=10)
ax.set_title('Mass Range Truncation Sensitivity', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_truncation.png'), dpi=200)
plt.close()

print("Figures generated.")

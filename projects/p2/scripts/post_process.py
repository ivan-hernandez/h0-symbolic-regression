"""
Post-process saved PySR results, generate plots.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob, pandas as pd

# Find hall of fame CSVs (check both project and legacy paths)
script_dir = os.path.dirname(os.path.abspath(__file__))
search_dirs = [
    os.path.join(script_dir, '..', 'analysis'),
    os.path.join(script_dir, '..', '..', 'analysis'),  # legacy: from ~/projects/p2, ../analysis = ~/projects/analysis
]
csvs = []
for d in search_dirs:
    csvs.extend(sorted(glob.glob(os.path.join(d, 'p2_sr_seed*', 'hall_of_fame.csv'))))
csvs = sorted(set(csvs))
    
print(f"Found {len(csvs)} PySR result files")

# Load data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
data = np.load(os.path.join(data_dir, 'gwtc3.npz'))
m1 = data['events']['m1']

# Reconstruct bins (matching p2_sr.py)
bins_per_dex = 5
min_m = 1.0
max_m = 200.0
nbins = int(np.ceil(np.log10(max_m/min_m) * bins_per_dex))
bins = np.logspace(np.log10(min_m), np.log10(max_m), nbins + 1)
counts, _ = np.histogram(m1, bins=bins)
bin_centers = np.sqrt(bins[:-1] * bins[1:])
keep = counts > 0
x = np.log10(bin_centers[keep])
y = np.log10(counts[keep].astype(float))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#e41a1c', '#377eb8', '#4daf4a']
labels = []

for i, csv_path in enumerate(csvs[:3]):
    df = pd.read_csv(csv_path)
    # Get best equation (lowest loss at max complexity)
    best_idx = df['Loss'].idxmin()
    eq = df.iloc[best_idx]['Equation']
    complexity = df.iloc[best_idx]['Complexity']
    loss = df.iloc[best_idx]['Loss']
    
    print(f"Seed {i}: complexity={complexity}, loss={loss:.5f}")
    print(f"  Equation: {eq}")
    
    # Parse seed from path
    seed = None
    for part in csv_path.split(os.sep):
        if 'seed' in part:
            seed = part.replace('p2_sr_seed', '')
    
    label = f'Seed {seed}' if seed else f'Run {i}'
    labels.append(label)
    
    # Evaluate
    m_grid = np.logspace(np.log10(min_m), np.log10(max_m), 300)
    xg = np.log10(m_grid)
    
    # Evaluate SR expression using simple parsing for common forms
    # The equation is in sympy format like (2.1528 - x0)*exp(x0*1.3787 - 0.3293/x0**2)/2.5948
    # We need to convert x0 -> xg and eval
    from sympy import sympify, lambdify
    try:
        expr = sympify(eq)
        f = lambdify('x0', expr, 'numpy')
        y_pred = f(xg)
        
        ax = axes[0]
        if i == 0:
            ax.errorbar(10**x, 10**y, yerr=10**y * (1.0/np.sqrt(counts[keep])/np.log(10)) * np.log(10), 
                       fmt='o', alpha=0.5, color='gray', label='Data')
        ax.plot(m_grid, 10**y_pred, '-', lw=2, color=colors[i], label=label)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('m1 (M$_\\odot$)')
        ax.set_ylabel('Count per bin')
        ax.set_title(f'Mass Distribution Fit (loss={loss:.3f})')
        ax.legend(); ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        # PDF
        y_vals = 10**y_pred
        norm = np.trapezoid(y_vals, m_grid)
        pdf_pred = y_vals / norm
        if i == 0:
            count_norm = np.trapezoid(counts.astype(float), bin_centers)
            pdf_data = counts / count_norm
            ax.step(bin_centers, pdf_data, where='mid', alpha=0.5, color='gray', label='Data')
        ax.plot(m_grid, pdf_pred, '-', lw=2, color=colors[i], label=label)
        ax.set_xlabel('m1 (M$_\\odot$)')
        ax.set_ylabel('PDF')
        ax.set_title('Probability Density')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 150)
        
    except Exception as e:
        print(f"  Error evaluating: {e}")

fig.tight_layout()
outpath = os.path.join(script_dir, '..', 'analysis', 'sr_results.png')
fig.savefig(outpath, dpi=150)
print(f"\nSaved {outpath}")

# Also do a quick comparison with Power Law + Peak
# Power law: p(m) ∝ m^alpha for m in [mmin, mmax]
# Power Law + Peak: includes Gaussian bump at ~35 Msun
print("\n=== Power Law + Peak Baseline ===")
print("LVK 2023 (arXiv:2111.03634):")
print("  alpha = -3.4 (power law slope)")
print("  mmin = 5.0 Msun, mmax = 87 Msun")
print("  Peak: Gaussian at mu=34 Msun, sigma=3.2 Msun, fraction=0.04")
print("  Smooth: delta_m = 4.8 Msun")

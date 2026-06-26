"""
Validate SR-discovered mass distributions.
Check for poles, singularities, overfitting.
Compare to Power Law + Peak baseline properly.
"""
import numpy as np
import sys, os, glob, pandas as pd
from sympy import sympify, lambdify

# Find all hall_of_fame CSVs
script_dir = os.path.dirname(os.path.abspath(__file__))
search_dirs = [
    os.path.join(script_dir, '..', 'analysis'),
    os.path.join(script_dir, '..', '..', 'analysis'),
]
csvs = []
for d in search_dirs:
    csvs.extend(sorted(glob.glob(os.path.join(d, 'p2_sr_seed*', 'hall_of_fame.csv'))))
csvs = sorted(set(csvs))
print(f"Found {len(csvs)} PySR result files")

# Load data
data = np.load(os.path.join(script_dir, '..', 'data', 'gwtc3.npz'))
m1 = data['events']['m1']

# For the histogram approach: Poisson log-likelihood
# log L = sum(counts * log(pred) - pred - log(counts!))
def poisson_log_likelihood(counts, pred):
    return np.sum(counts * np.log(pred + 1e-10) - pred)

# For individual events: log likelihood = sum(log(pdf(m_i)))
def individual_log_likelihood(pdf_vals):
    return np.sum(np.log(pdf_vals + 1e-10))

# Binning
bins_per_dex = 5
min_m, max_m = 1.0, 200.0
nbins = int(np.ceil(np.log10(max_m/min_m) * bins_per_dex))
bins = np.logspace(np.log10(min_m), np.log10(max_m), nbins + 1)
counts, _ = np.histogram(m1, bins=bins)
bin_centers = np.sqrt(bins[:-1] * bins[1:])

results = []
for csv_path in csvs:
    df = pd.read_csv(csv_path)
    best_idx = df['Loss'].idxmin()
    eq = df.iloc[best_idx]['Equation']
    comp = df.iloc[best_idx]['Complexity']
    loss = df.iloc[best_idx]['Loss']
    
    seed = ''.join([c for c in csv_path.split('/')[-2] if c.isdigit()])
    
    # Evaluate
    try:
        expr = sympify(eq)
        f = lambdify('x0', expr, 'numpy')
        
        # Check across the mass range
        x_test = np.linspace(0.01, 2.5, 1000)
        y_test = f(x_test)
        
        # Check for NaN/Inf
        has_nan = np.any(~np.isfinite(y_test))
        # Check for negative values
        has_negative = np.any(y_test < -1)
        # Count non-finite regions
        bad = ~np.isfinite(y_test)
        pole_regions = []
        for i in range(1, len(bad)):
            if bad[i] and not bad[i-1]:
                pole_regions.append((x_test[i], 'start'))
            if not bad[i] and bad[i-1]:
                pole_regions.append((x_test[i-1], 'end'))
        
        # Evaluate on bin grid
        m_grid = np.logspace(np.log10(min_m), np.log10(max_m), 500)
        y_pred = f(np.log10(m_grid))
        
        # Check for pole near primary mass range
        m_pole = None
        if has_nan:
            # Find where the pole is
            pole_idx = np.where(~np.isfinite(y_test))[0]
            if len(pole_idx) > 0:
                mid_pole = x_test[len(pole_idx)//2]
                m_pole = 10**mid_pole
                print(f"  Seed {seed}: POLE at m~{m_pole:.1f} M_sun (x0={mid_pole:.4f})")
        
        results.append({
            'seed': seed,
            'equation': eq,
            'complexity': comp,
            'loss': loss,
            'has_pole': has_nan or has_negative,
            'm_pole': m_pole,
            'f': f,
            'y_pred': y_pred,
        })
        print(f"  Seed {seed}: loss={loss:.5f}, comp={comp}, pole={has_nan or has_negative}")
        
    except Exception as e:
        print(f"  Seed {seed}: evaluation error: {e}")

print(f"\n=== Validation Results ===")
for r in results:
    status = 'REJECTED' if r['has_pole'] else 'OK'
    print(f"  Seed {r['seed']}: loss={r['loss']:.5f} {status}")
    if 'm_pole' in r and r['m_pole']:
        print(f"    Pole at m={r['m_pole']:.1f} M_sun — within primary mass range!")
    
# Compute PDF for accepted models
print(f"\n=== PDF Quality ===")
valid = [r for r in results if not r['has_pole']]
if valid:
    for r in valid:
        y_pred = r['y_pred']
        pdf = 10**y_pred  # count density in log10(m) space
        # Normalize properly
        pdf_norm = np.trapezoid(pdf, np.log10(m_grid))
        pdf = pdf / pdf_norm  # now it's a PDF in log10(m) space
        # Transform to PDF in linear m space
        # p(m) dm = p(log m) d(log m) = p(log m) / (m * ln(10)) dm
        pdf_linear = pdf / (m_grid * np.log(10))
        pdf_linear /= np.trapezoid(pdf_linear, m_grid)
        
        # Log-likelihood on individual events
        pdf_at_events = np.interp(m1, m_grid, pdf_linear)
        ll = individual_log_likelihood(pdf_at_events)
        print(f"  Seed {r['seed']}: logL={ll:.1f}, BIC={-2*ll + r['complexity']*np.log(len(m1)):.1f}")

# Power Law + Peak baseline
print(f"\n=== Power Law + Peak Baseline (LVK 2023) ===")
# Model: p(m) = [(1-lambda)*P(m) + lambda*G(m)] * S(m)
# P(m) ∝ m^alpha for m in [mmin, mmax]
# G(m) = N(mu, sigma)
# S(m) = smooth taper at low mass
alpha = -3.4
mmin, mmax = 5.0, 87.0
mu_peak, sigma_peak = 34.0, 3.2
lambda_peak = 0.04
delta_m = 4.8

m = m_grid
# Power law
pl = m**alpha
pl[(m < mmin) | (m > mmax)] = 0

# Smooth taper (logistic)
def smooth(m, low, high, delta):
    return 1 / (1 + np.exp(-(m - low)/delta)) * 1 / (1 + np.exp((m - high)/delta))

# Gaussian peak
gauss = np.exp(-0.5 * ((m - mu_peak) / sigma_peak)**2)
gauss /= np.sqrt(2*np.pi) * sigma_peak

# Combined
model = (1-lambda_peak) * pl * smooth(m, mmin, mmax, delta_m) + lambda_peak * gauss
model_norm = np.trapezoid(model, m)
model = model / model_norm

# Log-likelihood
pdf_at_events = np.interp(m1, m, model)
ll_plpeak = individual_log_likelihood(pdf_at_events)
n_params = 7  # alpha, mmin, mmax, mu_peak, sigma_peak, lambda_peak, delta_m
bic_plpeak = -2*ll_plpeak + n_params * np.log(len(m1))
print(f"  logL={ll_plpeak:.1f}, BIC={bic_plpeak:.1f}")

# Simple power law (no peak) for comparison
pl_simple = m**alpha * smooth(m, mmin, mmax, delta_m)
pl_simple_norm = np.trapezoid(pl_simple, m)
pl_simple = pl_simple / pl_simple_norm
pdf_at_events_simple = np.interp(m1, m, pl_simple)
ll_simple = individual_log_likelihood(pdf_at_events_simple)
n_params_simple = 4
bic_simple = -2*ll_simple + n_params_simple * np.log(len(m1))
print(f"Simple Power Law (no peak): logL={ll_simple:.1f}, BIC={bic_simple:.1f}")

print(f"\n=== Assessment ===")
print(f"Best SR model vs Power Law+Peak:")
if valid:
    for r in valid:
        y_pred = r['y_pred']
        pdf = 10**y_pred
        pdf_norm = np.trapezoid(pdf, np.log10(m_grid))
        pdf = pdf / pdf_norm
        pdf_linear = pdf / (m_grid * np.log(10))
        pdf_linear /= np.trapezoid(pdf_linear, m_grid)
        pdf_at_events = np.interp(m1, m_grid, pdf_linear)
        ll = individual_log_likelihood(pdf_at_events)
        bic = -2*ll + r['complexity'] * np.log(len(m1))
        delta_bic = bic - bic_plpeak
        print(f"  Seed {r['seed']}: ΔBIC(PL+Peak) = {delta_bic:.1f}")
        if delta_bic < -10:
            print(f"    → Strongly preferred over PL+Peak (ΔBIC < -10)")
        elif delta_bic < -2:
            print(f"    → Moderately preferred")
        elif delta_bic > 10:
            print(f"    → Strongly disfavored")
        elif delta_bic > 2:
            print(f"    → Moderately disfavored")
        else:
            print(f"    → Comparable")

# Crap-or-Worthwhile
print(f"\n=== Crap-or-Worthwhile Test ===")
print(f"If SR discovers simpler functional form for mass distribution than PL+Peak,")
print(f"  does this change the physical interpretation of BBH formation?")
print(f"  → An SR form with 4-5 parameters that matches PL+Peak (7 params) is interesting")
print(f"  → But if it's just re-fitting the same shape, it's practice")
print(f"Key novelty question: does the functional form suggest new physics?")

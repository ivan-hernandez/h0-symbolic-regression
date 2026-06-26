"""
Clean validation: check SR forms vs PL+Peak.
Handle log10(counts) → PDF conversion properly.
"""
import numpy as np, os, pandas as pd, glob
from sympy import sympify, lambdify

script_dir = os.path.dirname(os.path.abspath(__file__))
search_dirs = [os.path.join(script_dir, '..', 'analysis'), os.path.join(script_dir, '..', '..', 'analysis')]
csvs = sorted(set(g for d in search_dirs for g in glob.glob(os.path.join(d, 'p2_sr_seed*', 'hall_of_fame.csv'))))
print(f"Found {len(csvs)} PySR result files")

data = np.load(os.path.join(script_dir, '..', 'data', 'gwtc3.npz'))
m1 = data['events']['m1']
print(f"N={len(m1)} events")

# Binning parameters
bins_per_dex = 5
min_m, max_m = 1.0, 200.0
nbins = int(np.ceil(np.log10(max_m/min_m) * bins_per_dex))
bins = np.logspace(np.log10(min_m), np.log10(max_m), nbins + 1)

# Evaluation grid
m_grid = np.logspace(np.log10(min_m), np.log10(max_m), 500)

for csv_path in csvs:
    df = pd.read_csv(csv_path)
    best_idx = df['Loss'].idxmin()
    eq = df.iloc[best_idx]['Equation']
    comp = df.iloc[best_idx]['Complexity']
    loss = df.iloc[best_idx]['Loss']
    seed = ''.join(c for c in csv_path.split('/')[-2] if c.isdigit())
    
    print(f"\nSeed {seed} (actual): complexity={comp}, loss={loss:.5f}")
    print(f"  Equation: {eq}")
    
    try:
        # Parse equation - handle square(x) -> x^2
        # PySR uses "square(x0)" which sympy might not know
        eq_clean = eq.replace('square(x0)', 'x0**2').replace('square(x0)', 'x0**2')
        expr = sympify(eq_clean)
        f = lambdify('x0', expr, 'numpy')
        
        # Check for poles in [0, 2.5] log10(m) range
        x_test = np.linspace(0, 2.5, 10000)
        y_test = f(x_test)
        
        # Check for actual singularities (not just negative)
        non_finite = ~np.isfinite(y_test)
        n_poles = non_finite.sum()
        
        if n_poles > 0:
            pole_locs = x_test[non_finite]
            print(f"  ⚠ {n_poles} pole(s) detected at x0 ≈ {pole_locs[:5]}")
            print(f"  → Corresponding m ≈ {10**pole_locs[:5]} M_sun")
            continue  # Skip pathological models
        
        # Evaluate on grid
        xg = np.log10(m_grid)
        y_pred = f(xg)  # log10(counts)
        
        # Compute predicted counts in bins (count density in log10(m))
        # dN = 10^y_pred * d(log10(m))
        # Then compare to actual counts
        dlogm = np.log10(bins[1]/bins[0])
        bin_mid_x = np.log10(np.sqrt(bins[:-1] * bins[1:]))
        pred_counts = 10**f(bin_mid_x) * dlogm
        
        actual_counts, _ = np.histogram(m1, bins=bins)
        
        # Poisson log-likelihood
        ll = np.sum(actual_counts * np.log(pred_counts + 1e-10) - pred_counts)
        bic = -2*ll + comp * np.log(len(m1))
        
        print(f"  Binned logL={ll:.1f}, BIC={bic:.1f}")
        
        # Convert to PDF for individual event comparison
        # dN/dlog(m) = 10^f
        # dN/dm = dN/dlog(m) * 1/(m*ln(10))
        dndlogm = 10**f(np.log10(m_grid))
        dndm = dndlogm / (m_grid * np.log(10))
        norm = np.trapezoid(dndm, m_grid)
        pdf = dndm / norm
        
        # Log-likelihood on individual events
        pdf_at_events = np.interp(m1, m_grid, pdf)
        ll_indiv = np.sum(np.log(pdf_at_events + 1e-10))
        bic_indiv = -2*ll_indiv + comp * np.log(len(m1))
        
        print(f"  Individual logL={ll_indiv:.1f}, BIC={bic_indiv:.1f}")
        
    except Exception as e:
        print(f"  Error: {e}")

# PL+Peak baseline
print(f"\n{'='*50}")
print(f"Power Law + Peak Baseline (LVK 2023)")
alpha = -3.4; mmin, mmax = 5.0, 87.0; mu_pk, sig_pk = 34.0, 3.2; lam_pk = 0.04; dm = 4.8
m = m_grid
pl = m**alpha
smooth = 1/(1+np.exp(-(m-mmin)/dm)) * 1/(1+np.exp((m-mmax)/dm))
gauss = np.exp(-0.5*((m-mu_pk)/sig_pk)**2) / (np.sqrt(2*np.pi)*sig_pk)
model = (1-lam_pk) * pl * smooth + lam_pk * gauss
model_norm = np.trapezoid(model, m)
model = model / model_norm

pdf_at_ev = np.interp(m1, m, model)
ll_pl = np.sum(np.log(pdf_at_ev + 1e-10))
bic_pl = -2*ll_pl + 7 * np.log(len(m1))
print(f"Individual logL={ll_pl:.1f}, BIC={bic_pl:.1f}")

# Simple power law (no peak)
print(f"\nSimple Power Law (no peak, α=-3.4):")
pl2 = m**alpha * smooth
pl2 /= np.trapezoid(pl2, m)
pdf_ev2 = np.interp(m1, m, pl2)
ll_simple = np.sum(np.log(pdf_ev2 + 1e-10))
bic_simple = -2*ll_simple + 4 * np.log(len(m1))
print(f"Individual logL={ll_simple:.1f}, BIC={bic_simple:.1f}")

print(f"\n{'='*50}")
print(f"Assessment: SR vs PL+Peak BIC difference")
print(f"ΔBIC = BIC_SR - BIC_PL+Peak")
print(f"Negative → SR preferred; Positive → PL+Peak preferred")
print(f"|ΔBIC| > 10 → strong preference")

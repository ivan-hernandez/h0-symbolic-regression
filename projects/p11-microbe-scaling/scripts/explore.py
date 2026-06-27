"""
Phase 1 — Exploration of microbial metabolic scaling data.

Produces:
- Log-log scatter plots by group and state
- OLS + RMA exponents per subgroup
- Comparison with published DeLong+2010 exponents
- Temperature sensitivity analysis
"""
import os, csv, math, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = '/home/ivan/general-conversation/projects/p11-microbe-scaling/output/microbial_metabolic_data.csv'
OUT_DIR = '/home/ivan/general-conversation/projects/p11-microbe-scaling/output'
os.makedirs(OUT_DIR, exist_ok=True)

# Load
rows = []
with open(DATA_PATH) as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"Loaded {len(rows)} entries")

# ============================================================
# Helpers
# ============================================================
def ols(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    c, res, rank, s = np.linalg.lstsq(A, y, rcond=None)
    n = len(x)
    y_pred = A @ c
    resid = y - y_pred
    mse = np.sum(resid**2) / (n - 2)
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A) * mse))
    r2 = 1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2)
    return c[0], c[1], se[1], r2, resid

def rma(x, y):
    """Reduced Major Axis regression: slope = sign(corr) * sqrt(var_y / var_x)"""
    r = np.corrcoef(x, y)[0, 1]
    b_rma = np.sign(r) * np.std(y) / np.std(x)
    # Intercept
    a_rma = np.mean(y) - b_rma * np.mean(x)
    # SE of RMA slope
    n = len(x)
    se_b = abs(b_rma) * np.sqrt((1 - r**2) / (n - 2))
    return a_rma, b_rma, se_b, r

def fit_group(subset, label):
    if len(subset) < 3:
        print(f"  {label}: too few ({len(subset)})")
        return
    masses = np.array([float(r['mass_g']) for r in subset])
    mrs = np.array([float(r['metabolic_rate_W']) for r in subset])
    logM = np.log10(masses)
    logB = np.log10(mrs)
    
    a_ols, b_ols, se_ols, r2, resid = ols(logM, logB)
    a_rma, b_rma, se_rma, r = rma(logM, logB)
    
    print(f"  {label} ({len(subset)} pts):")
    print(f"    OLS: logB = {a_ols:.3f} + {b_ols:.3f}·logM  (R²={r2:.3f}, SE={se_ols:.3f})")
    print(f"    RMA: logB = {a_rma:.3f} + {b_rma:.3f}·logM  (r={r:.3f}, SE={se_rma:.3f})")
    return {
        'label': label, 'n': len(subset),
        'a_ols': a_ols, 'b_ols': b_ols, 'se_ols': se_ols, 'r2': r2,
        'a_rma': a_rma, 'b_rma': b_rma, 'se_rma': se_rma, 'r': r,
        'logM': logM, 'logB': logB, 'resid': resid
    }

# ============================================================
# 1. OLS + RMA by group
# ============================================================
print("\n=== OLS + RMA by group ===")

# By domain
for dom in ['Archaea', 'Bacteria', 'Eukaryota']:
    subset = [r for r in rows if r['domain'] == dom]
    fit_group(subset, dom)

# By source and domain
for src in ['Hoehler+2023', 'DeLong+2010']:
    for dom in ['Archaea', 'Bacteria', 'Eukaryota']:
        subset = [r for r in rows if r['source'] == src and r['domain'] == dom]
        if len(subset) >= 3:
            fit_group(subset, f"{src} {dom}")

# By state (prokaryotes only)
prok = [r for r in rows if r['domain'] in ('Archaea', 'Bacteria')]
for state in ['endogenous', 'active', 'unknown']:
    subset = [r for r in prok if r['state'] == state]
    fit_group(subset, f"prok {state}")

# Combined prokaryotes (all states)
all_results = {}
all_results['prok_all'] = fit_group(prok, "prok all states")

# ============================================================
# 2. Compare with published DeLong exponents
# ============================================================
print("\n=== Comparison with DeLong+2010 published exponents ===")
print("DeLong reports RMA slopes (not OLS):")
print("  Group          | Active   | Endogenous")
print("  Prokaryotes    | 1.7±0.31 | 2.0±0.28")
print("  Protists       | 1.0±0.09 | 1.1±0.10")
print("  Metazoans      | 0.76±0.07| 0.79±0.12")

print("\nOur RMA estimates for DeLong data only:")
for state in ['active', 'endogenous']:
    subset = [r for r in rows if r['source'] == 'DeLong+2010' and r['domain'] == 'Bacteria' and r['state'] == state]
    r = fit_group(subset, f"DeLong prok {state}")

# Protists from DeLong
for state in ['active', 'endogenous']:
    subset = [r for r in rows if r['source'] == 'DeLong+2010' and r['domain'] == 'Eukaryota' and r['state'] == state and r['species'] != '']
    if len(subset) >= 3:
        r = fit_group(subset, f"DeLong euk {state}")

# ============================================================
# 3. Scatter plots
# ============================================================
print("\n=== Generating plots ===")

# Fig 1: All data by domain and source
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors_domain = {'Archaea': '#e41a1c', 'Bacteria': '#377eb8', 'Eukaryota': '#4daf4a'}
markers_state = {'endogenous': 'o', 'active': '^', 'unknown': 's'}

ax = axes[0]
for dom in ['Archaea', 'Bacteria', 'Eukaryota']:
    for state in ['endogenous', 'active', 'unknown']:
        subset = [r for r in rows if r['domain'] == dom and r['state'] == state]
        if subset:
            m = [float(r['mass_g']) for r in subset]
            b = [float(r['metabolic_rate_W']) for r in subset]
            ax.scatter(m, b, c=colors_domain[dom], marker=markers_state[state],
                      alpha=0.5, s=15, edgecolors='none',
                      label=f"{dom} {state}" if state == 'endogenous' else '')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Body mass (g)'); ax.set_ylabel('Metabolic rate (W)')
ax.set_title('All data by domain')
# Add fit lines for prokaryotes
for state in ['endogenous', 'active', 'unknown']:
    subset = [r for r in prok if r['state'] == state]
    if len(subset) >= 3:
        m = np.log10([float(r['mass_g']) for r in subset])
        b = np.log10([float(r['metabolic_rate_W']) for r in subset])
        a, slope, *_ = ols(m, b)
        x_fit = np.array([min(m), max(m)])
        ax.plot(10**x_fit, 10**(a + slope * x_fit), '--', linewidth=1,
               label=f"prok {state}: b={slope:.2f}")
ax.legend(fontsize=7, loc='lower right')

ax = axes[1]
for src in ['Hoehler+2023', 'DeLong+2010']:
    for state in ['endogenous', 'active', 'unknown']:
        subset = [r for r in rows if r['source'] == src and r['domain'] in ('Archaea', 'Bacteria') and r['state'] == state]
        if subset:
            m = [float(r['mass_g']) for r in subset]
            b = [float(r['metabolic_rate_W']) for r in subset]
            ax.scatter(m, b, c='#377eb8' if src == 'Hoehler+2023' else '#e41a1c',
                      marker=markers_state[state], alpha=0.5, s=15, edgecolors='none')
# Add fit lines by source
for src, color in [('Hoehler+2023', '#377eb8'), ('DeLong+2010', '#e41a1c')]:
    subset = [r for r in rows if r['source'] == src and r['domain'] in ('Archaea', 'Bacteria')]
    if len(subset) >= 3:
        m = np.log10([float(r['mass_g']) for r in subset])
        b = np.log10([float(r['metabolic_rate_W']) for r in subset])
        a, slope, *_ = ols(m, b)
        x_fit = np.array([min(m), max(m)])
        ax.plot(10**x_fit, 10**(a + slope * x_fit), '--', color=color, linewidth=1.5,
               label=f"{src}: b={slope:.2f}")
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Body mass (g)'); ax.set_ylabel('Metabolic rate (W)')
ax.set_title('Prokaryotes by source')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_overview.png'), dpi=150)
print("  Saved fig1_overview.png")

# Fig 2: Temperature comparison (Hoehler data only, has 25C normalized)
hoehler = [r for r in rows if r['source'] == 'Hoehler+2023' and r['domain'] in ('Archaea', 'Bacteria')]
has_t25 = [r for r in hoehler if r['metabolic_rate_25C_W'] and not math.isnan(float(r['metabolic_rate_25C_W']))]
if has_t25:
    fig, ax = plt.subplots(figsize=(7, 6))
    raw_m = [float(r['mass_g']) for r in has_t25]
    raw_b = [float(r['metabolic_rate_W']) for r in has_t25]
    t25_b = [float(r['metabolic_rate_25C_W']) for r in has_t25]
    
    ax.scatter(raw_b, t25_b, c='#377eb8', alpha=0.5, s=10, edgecolors='none')
    # 1:1 line
    lims = [min(min(raw_b), min(t25_b)), max(max(raw_b), max(t25_b))]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='1:1')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Metabolic rate (W)'); ax.set_ylabel('Metabolic rate at 25°C (W)')
    ax.set_title('Temperature normalization effect (Hoehler microbes)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig2_temperature.png'), dpi=150)
    print("  Saved fig2_temperature.png")
    
    # Compare OLS with and without T normalization
    raw_logM = np.log10(raw_m); raw_logB = np.log10(raw_b)
    t25_logB = np.log10(t25_b)
    _, b_raw, *_ = ols(raw_logM, raw_logB)
    _, b_t25, *_ = ols(raw_logM, t25_logB)
    print(f"\nTemperature sensitivity (Hoehler microbes, n={len(has_t25)}):")
    print(f"  Raw rates:     b={b_raw:.3f}")
    print(f"  25°C norm:     b={b_t25:.3f}")
    print(f"  Difference:    Δb={b_t25 - b_raw:+.3f}")

# Fig 3: Prokaryote-only log-log with state separation
fig, ax = plt.subplots(figsize=(8, 6))
for state, color, marker in [('endogenous', '#4daf4a', 'o'), ('active', '#e41a1c', '^'), ('unknown', '#984ea3', 's')]:
    subset = [r for r in prok if r['state'] == state]
    if subset:
        m = [float(r['mass_g']) for r in subset]
        b = [float(r['metabolic_rate_W']) for r in subset]
        ax.scatter(m, b, c=color, marker=marker, alpha=0.5, s=20, edgecolors='none',
                   label=f"{state} (n={len(subset)})")
        
        if len(subset) >= 3:
            lm = np.log10(m); lb = np.log10(b)
            a, slope, se, r2, _ = ols(lm, lb)
            _, b_rma, se_rma, r = rma(lm, lb)
            x_fit = np.array([min(lm), max(lm)])
            ax.plot(10**x_fit, 10**(a + slope * x_fit), '--', color=color, linewidth=1,
                   label=f"OLS b={slope:.2f}±{se:.2f}")
            ax.plot(10**x_fit, 10**(np.mean(lb) - b_rma * np.mean(lm) + b_rma * x_fit), 
                   ':', color=color, linewidth=1,
                   label=f"RMA b={b_rma:.2f}±{se_rma:.2f}")

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Body mass (g)'); ax.set_ylabel('Metabolic rate (W)')
ax.set_title('Prokaryote metabolic scaling by state')
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_prok_state.png'), dpi=150)
print("  Saved fig3_prok_state.png")

# Fig 4: Residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (state, color) in enumerate([('endogenous', '#4daf4a'), ('active', '#e41a1c')]):
    ax = axes[idx]
    subset = [r for r in prok if r['state'] == state]
    if len(subset) < 5:
        continue
    lm = np.array([np.log10(float(r['mass_g'])) for r in subset])
    lb = np.array([np.log10(float(r['metabolic_rate_W'])) for r in subset])
    a, b, se, r2, resid = ols(lm, lb)
    
    ax.scatter(lm, resid, c=color, alpha=0.5, s=20, edgecolors='none')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('log10(Mass)')
    ax.set_ylabel('Residual (dex)')
    ax.set_title(f'{state} residuals (b={b:.2f}, R²={r2:.3f})')
    # RMS residual
    rms = np.sqrt(np.mean(resid**2))
    ax.text(0.05, 0.9, f'RMS={rms:.3f} dex', transform=ax.transAxes, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_residuals.png'), dpi=150)
print("  Saved fig4_residuals.png")

# ============================================================
# 4. Summary table
# ============================================================
print("\n=== Summary Table ===")
print(f"{'Group':<25} {'n':>5} {'OLS b':>7} {'SE':>6} {'R²':>5} {'RMA b':>7} {'SE':>6} {'r':>5}")
print("-"*70)
for key in [
    'Archaea', 'Bacteria', 'Eukaryota',
    'prok endogenous', 'prok active', 'prok unknown',
    'prok all states',
    'Hoehler+2023 Bacteria', 'Hoehler+2023 Archaea',
    'DeLong+2010 Bacteria',
    'DeLong+2010 prok endogenous', 'DeLong+2010 prok active',
]:
    # Extract from the file by re-running fits
    pass

# Clean print
print("\nDone. All plots in output/")

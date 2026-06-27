"""
Phase 1: Explore the metabolic rate vs body mass relationship
using the AnimalTraits database (Herberstein+2022, Scientific Data).
"""
import os, sys, csv, statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'observations.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

ENDOTHERM_CLASSES = {'Mammalia', 'Aves'}
ECTO_THERM_CLASSES = {'Reptilia', 'Amphibia', 'Insecta', 'Arachnida', 'Malacostraca', 'Chilopoda'}

def load_data():
    rows = []
    with open(DATA_PATH, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mr = row.get('metabolic rate', '').strip()
            mass = row.get('body mass', '').strip()
            if not mr or mr == 'NA' or not mass or mass == 'NA':
                continue
            try:
                mr_f = float(mr)
                mass_f = float(mass)
            except (ValueError, TypeError):
                continue
            if mr_f <= 0 or mass_f <= 0:
                continue
            method = row.get('metabolic rate - method', '').strip().lower()
            cls = row.get('class', '')
            phylum = row.get('phylum', '')
            order = row.get('order', '')
            family = row.get('family', '')
            species = row.get('species', '')
            rows.append({
                'mr': mr_f, 'mass': mass_f,
                'log_mr': np.log10(mr_f), 'log_mass': np.log10(mass_f),
                'class': cls, 'phylum': phylum,
                'order': order, 'family': family, 'species': species,
                'method': method
            })
    return rows

def classify_thermo(cls):
    if cls in ENDOTHERM_CLASSES:
        return 'endotherm'
    elif cls in ECTO_THERM_CLASSES:
        return 'ectotherm'
    return 'unknown'

def power_law_fit(log_mass, log_mr):
    """OLS fit: log10(MR) = a + b * log10(mass), return a, b, chi2, r2"""
    A = np.vstack([np.ones_like(log_mass), log_mass]).T
    coeffs, resid, rank, s = np.linalg.lstsq(A, log_mr, rcond=None)
    a, b = coeffs[0], coeffs[1]
    pred = a + b * log_mass
    chi2 = np.sum((log_mr - pred)**2)
    ss_res = np.sum((log_mr - pred)**2)
    ss_tot = np.sum((log_mr - np.mean(log_mr))**2)
    r2 = 1 - ss_res / ss_tot
    return a, b, chi2, r2

def quadratic_fit(log_mass, log_mr):
    """Quadratic fit: log10(MR) = a + b*log10(mass) + c*(log10(mass))^2"""
    A = np.vstack([np.ones_like(log_mass), log_mass, log_mass**2]).T
    coeffs, resid, rank, s = np.linalg.lstsq(A, log_mr, rcond=None)
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    pred = a + b * log_mass + c * log_mass**2
    chi2 = np.sum((log_mr - pred)**2)
    ss_res = np.sum((log_mr - pred)**2)
    ss_tot = np.sum((log_mr - np.mean(log_mr))**2)
    r2 = 1 - ss_res / ss_tot
    return a, b, c, chi2, r2

def main():
    rows = load_data()
    print(f"Total paired observations: {len(rows)}")

    # Thermoregulation split
    for r in rows:
        r['thermo'] = classify_thermo(r['class'])

    endo = [r for r in rows if r['thermo'] == 'endotherm']
    ecto = [r for r in rows if r['thermo'] == 'ectotherm']
    print(f"  Endotherms: {len(endo)}")
    print(f"  Ectotherms: {len(ecto)}")
    print(f"  Unknown:    {len(rows) - len(endo) - len(ecto)}")

    # Method breakdown
    methods = {}
    for r in rows:
        methods[r['method']] = methods.get(r['method'], 0) + 1
    print("\nMethod breakdown:")
    for m, c in sorted(methods.items()):
        print(f"  {m}: {c}")

    # Class breakdown
    classes = {}
    for r in rows:
        classes[r['class']] = classes.get(r['class'], 0) + 1
    print("\nClass breakdown:")
    for c, n in sorted(classes.items()):
        print(f"  {c}: {n}")

    # Mass range
    masses = np.array([r['mass'] for r in rows])
    mrs = np.array([r['mr'] for r in rows])
    print(f"\nMass range: {masses.min():.2e} to {masses.max():.2e} kg ({np.log10(masses.max()/masses.min()):.1f} dex)")
    print(f"BMR range: {mrs.min():.2e} to {mrs.max():.2e} W")

    # Full dataset: power law fit
    log_mass = np.array([r['log_mass'] for r in rows])
    log_mr = np.array([r['log_mr'] for r in rows])
    a, b, chi2, r2 = power_law_fit(log_mass, log_mr)
    print(f"\n=== Power Law Fit (all data, N={len(rows)}) ===")
    print(f"  log10(MR) = {a:.4f} + {b:.4f} * log10(M)")
    print(f"  MR ∝ M^{b:.4f}")
    print(f"  R² = {r2:.4f}, χ² = {chi2:.2f}")
    print(f"  Predicted exponent: 2/3={2/3:.4f}, 3/4={3/4:.4f}")

    # Full dataset: quadratic fit
    a_q, b_q, c_q, chi2_q, r2_q = quadratic_fit(log_mass, log_mr)
    print(f"\n=== Quadratic Fit (all data, N={len(rows)}) ===")
    print(f"  log10(MR) = {a_q:.4f} + {b_q:.4f}*log10(M) + {c_q:.4f}*log10(M)²")
    delta_chi2 = chi2 - chi2_q
    print(f"  Δχ² vs power law: {delta_chi2:.2f} (Δdof=1)")
    print(f"  R² = {r2_q:.4f}")

    # Endotherms only
    log_mass_endo = np.array([r['log_mass'] for r in endo])
    log_mr_endo = np.array([r['log_mr'] for r in endo])
    a_e, b_e, chi2_e, r2_e = power_law_fit(log_mass_endo, log_mr_endo)
    print(f"\n=== Power Law Fit (endotherms, N={len(endo)}) ===")
    print(f"  log10(MR) = {a_e:.4f} + {b_e:.4f} * log10(M)")
    print(f"  MR ∝ M^{b_e:.4f}, R² = {r2_e:.4f}")

    # Ectotherms only
    log_mass_ecto = np.array([r['log_mass'] for r in ecto])
    log_mr_ecto = np.array([r['log_mr'] for r in ecto])
    a_c, b_c, chi2_c, r2_c = power_law_fit(log_mass_ecto, log_mr_ecto)
    print(f"\n=== Power Law Fit (ectotherms, N={len(ecto)}) ===")
    print(f"  log10(MR) = {a_c:.4f} + {b_c:.4f} * log10(M)")
    print(f"  MR ∝ M^{b_c:.4f}, R² = {r2_c:.4f}")

    # Mammals only
    mammals = [r for r in rows if r['class'] == 'Mammalia']
    log_mass_mam = np.array([r['log_mass'] for r in mammals])
    log_mr_mam = np.array([r['log_mr'] for r in mammals])
    if len(mammals) > 10:
        a_m, b_m, chi2_m, r2_m = power_law_fit(log_mass_mam, log_mr_mam)
        print(f"\n=== Power Law Fit (Mammalia, N={len(mammals)}) ===")
        print(f"  log10(MR) = {a_m:.4f} + {b_m:.4f} * log10(M)")
        print(f"  MR ∝ M^{b_m:.4f}, R² = {r2_m:.4f}")

    # Birds only
    birds = [r for r in rows if r['class'] == 'Aves']
    log_mass_bird = np.array([r['log_mass'] for r in birds])
    log_mr_bird = np.array([r['log_mr'] for r in birds])
    if len(birds) > 10:
        a_bi, b_bi, chi2_bi, r2_bi = power_law_fit(log_mass_bird, log_mr_bird)
        print(f"\n=== Power Law Fit (Aves, N={len(birds)}) ===")
        print(f"  log10(MR) = {a_bi:.4f} + {b_bi:.4f} * log10(M)")
        print(f"  MR ∝ M^{b_bi:.4f}, R² = {r2_bi:.4f}")

    # Insects only
    insects = [r for r in rows if r['class'] == 'Insecta']
    log_mass_ins = np.array([r['log_mass'] for r in insects])
    log_mr_ins = np.array([r['log_mr'] for r in insects])
    if len(insects) > 10:
        a_i, b_i, chi2_i, r2_i = power_law_fit(log_mass_ins, log_mr_ins)
        print(f"\n=== Power Law Fit (Insecta, N={len(insects)}) ===")
        print(f"  log10(MR) = {a_i:.4f} + {b_i:.4f} * log10(M)")
        print(f"  MR ∝ M^{b_i:.4f}, R² = {r2_i:.4f}")

    #====================
    # Plots
    #====================

    # 1. Log-log scatter all data
    fig, ax = plt.subplots(figsize=(9, 7))
    endo_mass = np.array([r['mass'] for r in endo])
    endo_mr = np.array([r['mr'] for r in endo])
    ecto_mass = np.array([r['mass'] for r in ecto])
    ecto_mr = np.array([r['mr'] for r in ecto])
    ax.scatter(ecto_mass, ecto_mr, c='C0', alpha=0.4, s=8, label=f'Ectotherms (N={len(ecto)})')
    ax.scatter(endo_mass, endo_mr, c='C3', alpha=0.4, s=8, label=f'Endotherms (N={len(endo)})')
    m_grid = np.logspace(-9, 3, 200)
    mr_pl = 10**(a + b * np.log10(m_grid))
    ax.plot(m_grid, mr_pl, 'k-', lw=2, label=f'PL: MR ∝ M^{b:.3f}')
    mr_q = 10**(a_q + b_q * np.log10(m_grid) + c_q * np.log10(m_grid)**2)
    ax.plot(m_grid, mr_q, 'k--', lw=1.5, label='Quadratic')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Body Mass (kg)', fontsize=12)
    ax.set_ylabel('Metabolic Rate (W)', fontsize=12)
    ax.set_title('Metabolic Rate vs Body Mass (AnimalTraits)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mr_vs_mass_scatter.png'), dpi=150)
    print(f"\nSaved mr_vs_mass_scatter.png")

    # 2. Residuals from power law
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    resid = log_mr - (a + b * log_mass)
    ax.scatter(log_mass, resid, c='gray', alpha=0.3, s=5)
    ax.axhline(0, color='k', lw=1)
    # Binned residuals
    bins = np.linspace(log_mass.min(), log_mass.max(), 30)
    bin_c = (bins[1:] + bins[:-1]) / 2
    bin_m = np.array([np.mean(resid[(log_mass >= bins[i]) & (log_mass < bins[i+1])]) for i in range(len(bins)-1)])
    bin_s = np.array([np.std(resid[(log_mass >= bins[i]) & (log_mass < bins[i+1])]) / np.sqrt(max(1, np.sum((log_mass >= bins[i]) & (log_mass < bins[i+1])))) for i in range(len(bins)-1)])
    keep = np.isfinite(bin_m)
    ax.errorbar(bin_c[keep], bin_m[keep], yerr=bin_s[keep], fmt='o-', color='k', lw=2)
    ax.set_xlabel('log10(Body Mass [kg])')
    ax.set_ylabel('Residual (dex)')
    ax.set_title(f'Power Law Residuals (b={b:.3f})')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    resid_q = log_mr - (a_q + b_q * log_mass + c_q * log_mass**2)
    ax.scatter(log_mass, resid_q, c='gray', alpha=0.3, s=5)
    ax.axhline(0, color='k', lw=1)
    bin_m_q = np.array([np.mean(resid_q[(log_mass >= bins[i]) & (log_mass < bins[i+1])]) for i in range(len(bins)-1)])
    bin_s_q = np.array([np.std(resid_q[(log_mass >= bins[i]) & (log_mass < bins[i+1])]) / np.sqrt(max(1, np.sum((log_mass >= bins[i]) & (log_mass < bins[i+1])))) for i in range(len(bins)-1)])
    ax.errorbar(bin_c[keep], bin_m_q[keep], yerr=bin_s_q[keep], fmt='o-', color='k', lw=2)
    ax.set_xlabel('log10(Body Mass [kg])')
    ax.set_ylabel('Residual (dex)')
    ax.set_title(f'Quadratic Residuals (c={c_q:.4f})')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'residuals.png'), dpi=150)
    print("Saved residuals.png")

    # 3. Endotherm vs ectotherm fits
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(ecto_mass, ecto_mr, c='C0', alpha=0.3, s=8, label=f'Ectotherms')
    ax.scatter(endo_mass, endo_mr, c='C3', alpha=0.3, s=8, label=f'Endotherms')
    mr_pl_endo = 10**(a_e + b_e * np.log10(m_grid))
    mr_pl_ecto = 10**(a_c + b_c * np.log10(m_grid))
    ax.plot(m_grid, mr_pl_endo, 'r-', lw=2, label=f'Endotherm: M^{b_e:.3f}')
    ax.plot(m_grid, mr_pl_ecto, 'b-', lw=2, label=f'Ectotherm: M^{b_c:.3f}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Body Mass (kg)')
    ax.set_ylabel('Metabolic Rate (W)')
    ax.set_title('Endotherm vs Ectotherm Allometry')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'endo_vs_ecto.png'), dpi=150)
    print("Saved endo_vs_ecto.png")

    # 4. Taxonomic breakdown
    fig, ax = plt.subplots(figsize=(10, 7))
    class_colors = {'Mammalia': 'C3', 'Aves': 'C1', 'Reptilia': 'C2',
                    'Insecta': 'C0', 'Arachnida': 'C4', 'Malacostraca': 'C5'}
    for cls_name, color in class_colors.items():
        subset = [r for r in rows if r['class'] == cls_name]
        if len(subset) < 5: continue
        m_sub = np.array([r['mass'] for r in subset])
        mr_sub = np.array([r['mr'] for r in subset])
        ax.scatter(m_sub, mr_sub, c=color, alpha=0.4, s=8, label=f'{cls_name} ({len(subset)})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Body Mass (kg)')
    ax.set_ylabel('Metabolic Rate (W)')
    ax.set_title('Metabolic Allometry by Class')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'by_class.png'), dpi=150)
    print("Saved by_class.png")

    # Save cleaned data for PySR
    data_out = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_data.npz')
    arr_dtype = [('mass', 'f8'), ('mr', 'f8'),
                 ('log_mass', 'f8'), ('log_mr', 'f8'),
                 ('thermo', 'U10'), ('class', 'U20'),
                 ('order', 'U30'), ('family', 'U30'),
                 ('species', 'U50'), ('method', 'U25')]
    arr = np.array([(r['mass'], r['mr'], r['log_mass'], r['log_mr'],
                     r['thermo'], r['class'], r['order'], r['family'],
                     r['species'], r['method']) for r in rows], dtype=arr_dtype)
    np.savez(data_out, data=arr)
    print(f"\nSaved {data_out} ({len(arr)} rows)")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY: Power Law Exponents")
    print("="*60)
    print(f"{'Group':<20} {'N':>6} {'b':>8} {'R²':>8}")
    print("-"*60)
    print(f"{'All':<20} {len(rows):>6} {b:>8.4f} {r2:>8.4f}")
    print(f"{'Endotherms':<20} {len(endo):>6} {b_e:>8.4f} {r2_e:>8.4f}")
    print(f"{'Ectotherms':<20} {len(ecto):>6} {b_c:>8.4f} {r2_c:>8.4f}")
    if len(mammals) > 10:
        print(f"{'Mammalia':<20} {len(mammals):>6} {b_m:>8.4f} {r2_m:>8.4f}")
    if len(birds) > 10:
        print(f"{'Aves':<20} {len(birds):>6} {b_bi:>8.4f} {r2_bi:>8.4f}")
    if len(insects) > 10:
        print(f"{'Insecta':<20} {len(insects):>6} {b_i:>8.4f} {r2_i:>8.4f}")
    print(f"\nQuadratic Δχ² vs PL: {delta_chi2:.2f} (Δdof=1)")

if __name__ == "__main__":
    main()

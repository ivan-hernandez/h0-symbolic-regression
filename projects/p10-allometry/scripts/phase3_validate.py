"""
Phase 3: Validation of Kleiber's Law SR results
- Bootstrap resampling (200 resamples) for CPX5 parameters
- Taxonomic holdout (leave-one-order-out cross-validation)
- Residual normality / phylogenetic signal test
"""
import os, sys, csv, random
import numpy as np
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
            cls = row.get('class', '')
            method = row.get('metabolic rate - method', '').strip().lower()
            order = row.get('order', '')
            rows.append({
                'mr': mr_f, 'mass': mass_f,
                'log_mr': np.log10(mr_f), 'log_mass': np.log10(mass_f),
                'class': cls, 'order': order, 'method': method,
            })
    return rows

def classify_thermo(cls):
    if cls in ENDOTHERM_CLASSES: return 'endotherm'
    elif cls in ECTO_THERM_CLASSES: return 'ectotherm'
    return 'unknown'

def power_law_coeffs(log_mass, log_mr):
    A = np.vstack([np.ones_like(log_mass), log_mass]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_mr, rcond=None)
    return coeffs[0], coeffs[1]  # intercept, slope

def power_law_loss(params, log_mass, log_mr):
    a, b = params
    pred = a + b * log_mass
    return np.mean((log_mr - pred)**2)

def main():
    rows = load_data()
    for r in rows:
        r['thermo'] = classify_thermo(r['class'])

    # Define subsets
    subsets = {
        'all': lambda r: True,
        'mammalia': lambda r: r['class'] == 'Mammalia',
        'mammal_bmr': lambda r: r['class'] == 'Mammalia' and r['method'] == 'basal metabolic rate',
        'aves': lambda r: r['class'] == 'Aves',
        'endotherm': lambda r: r['thermo'] == 'endotherm',
        'ectotherm': lambda r: r['thermo'] == 'ectotherm',
        'insecta': lambda r: r['class'] == 'Insecta',
    }

    # =====================================================
    # 1. Bootstrap: 200 resamples, CPX5 power law
    # =====================================================
    print("=" * 70)
    print("BOOTSTRAP RESULTS (200 resamples, CPX5 power law)")
    print("=" * 70)
    print(f"{'Subset':<15} {'N':>5} {'a':>8} {'b':>8} {'σ_a':>8} {'σ_b':>8} {'b_2.5':>8} {'b_97.5':>8}")
    print("-" * 70)

    bootstrap_results = {}
    for sname, sfilter in subsets.items():
        subset = [r for r in rows if sfilter(r)]
        if len(subset) < 10:
            continue
        log_mass = np.array([r['log_mass'] for r in subset])
        log_mr = np.array([r['log_mr'] for r in subset])
        n = len(log_mass)

        # Point estimate
        a0, b0 = power_law_coeffs(log_mass, log_mr)

        # Bootstrap
        n_boot = 200
        boot_a = np.zeros(n_boot)
        boot_b = np.zeros(n_boot)
        for i in range(n_boot):
            idx = np.random.randint(0, n, n)
            bs_mass = log_mass[idx]
            bs_mr = log_mr[idx]
            boot_a[i], boot_b[i] = power_law_coeffs(bs_mass, bs_mr)

        a_mean, b_mean = np.mean(boot_a), np.mean(boot_b)
        a_std, b_std = np.std(boot_a), np.std(boot_b)
        b_lo, b_hi = np.percentile(boot_b, [2.5, 97.5])

        bootstrap_results[sname] = {
            'a': a_mean, 'b': b_mean,
            'a_std': a_std, 'b_std': b_std,
            'b_ci': (b_lo, b_hi),
            'n': n
        }

        print(f"{sname:<15} {n:>5} {a_mean:>8.4f} {b_mean:>8.4f} {a_std:>8.4f} {b_std:>8.4f} {b_lo:>8.4f} {b_hi:>8.4f}")

    # =====================================================
    # 2. Taxonomic holdout: leave-one-order-out
    # =====================================================
    print("\n" + "=" * 70)
    print("TAXONOMIC HOLDOUT (leave-one-order-out)")
    print("=" * 70)

    subsets_ordered = ['mammalia', 'aves', 'insecta']
    for sname in subsets_ordered:
        sfilter = subsets[sname]
        subset = [r for r in rows if sfilter(r)]
        if len(subset) < 20:
            continue

        # Get orders with enough species
        orders = {}
        for r in subset:
            orders[r['order']] = orders.get(r['order'], 0) + 1
        major_orders = [o for o, c in orders.items() if c >= 5]

        if len(major_orders) < 2:
            continue

        pred_errors = []
        held_out_names = []
        for held_order in major_orders:
            train = [r for r in subset if r['order'] != held_order]
            test = [r for r in subset if r['order'] == held_order]
            if len(train) < 10 or len(test) < 3:
                continue
            x_train = np.array([r['log_mass'] for r in train])
            y_train = np.array([r['log_mr'] for r in train])
            x_test = np.array([r['log_mass'] for r in test])
            y_test = np.array([r['log_mr'] for r in test])

            a, b = power_law_coeffs(x_train, y_train)
            pred = a + b * x_test
            rmse = np.sqrt(np.mean((y_test - pred)**2))
            pred_errors.append(rmse)
            held_out_names.append(held_order)

        if pred_errors:
            mean_rmse = np.mean(pred_errors)
            print(f"{sname:<15} N_orders={len(major_orders):>2} "
                  f"Mean holdout RMSE={mean_rmse:.4f} "
                  f"(range {min(pred_errors):.4f}–{max(pred_errors):.4f})")

    # =====================================================
    # 3. Residual analysis: phylogenetic signal
    # =====================================================
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS")
    print("=" * 70)

    for sname, sfilter in [('mammalia', subsets['mammalia']), ('aves', subsets['aves'])]:
        subset = [r for r in rows if sfilter(r)]
        if len(subset) < 20:
            continue
        log_mass = np.array([r['log_mass'] for r in subset])
        log_mr = np.array([r['log_mr'] for r in subset])
        a, b = power_law_coeffs(log_mass, log_mr)
        resid = log_mr - (a + b * log_mass)

        # Check if residuals correlate with order (simple ANOVA-like)
        orders = {}
        for r in subset:
            o = r['order']
            if o not in orders:
                orders[o] = []
        for i, r in enumerate(subset):
            orders[r['order']].append(resid[i])

        # Mean residual per order
        order_means = {o: np.mean(v) for o, v in orders.items() if len(v) >= 3}
        if order_means:
            vals = list(order_means.values())
            spread = np.std(vals)
            print(f"{sname:<15} N_orders={len(order_means):>2} "
                  f"σ_order_means={spread:.4f} dex "
                  f"(range {min(vals):+.4f} to {max(vals):+.4f})")

    # =====================================================
    # 4. Test specific exponents against literature
    # =====================================================
    print("\n" + "=" * 70)
    print("LITERATURE COMPARISON (mammal exponent)")
    print("=" * 70)
    mam = bootstrap_results.get('mammalia', {})
    bmr = bootstrap_results.get('mammal_bmr', {})

    if mam:
        b = mam['b']
        ci = mam['b_ci']
        print(f"Mammalia:   b = {b:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  vs Kleiber (3/4 = 0.750):  {'REJECTED' if 0.75 < ci[0] or 0.75 > ci[1] else 'consistent'}")
        print(f"  vs Surface law (2/3 = 0.667): {'REJECTED' if 0.667 < ci[0] or 0.667 > ci[1] else 'consistent'}")
    if bmr:
        b = bmr['b']
        ci = bmr['b_ci']
        print(f"Mammal BMR: b = {b:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  vs Kleiber (3/4 = 0.750):  {'REJECTED' if 0.75 < ci[0] or 0.75 > ci[1] else 'consistent'}")
        print(f"  vs Surface law (2/3 = 0.667): {'REJECTED' if 0.667 < ci[0] or 0.667 > ci[1] else 'consistent'}")


if __name__ == '__main__':
    main()

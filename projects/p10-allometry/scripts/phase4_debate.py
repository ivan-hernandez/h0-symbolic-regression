"""
Phase 4: Adversarial Debate — stress-test the power law result
"""
import os, sys, csv, random
import numpy as np
from scipy.optimize import minimize
from scipy import stats

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'observations.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

ENDOTHERM_CLASSES = {'Mammalia', 'Aves'}

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
            rows.append({
                'mr': mr_f, 'mass': mass_f,
                'log_mr': np.log10(mr_f), 'log_mass': np.log10(mass_f),
                'class': row.get('class', ''),
                'order': row.get('order', ''),
                'family': row.get('family', ''),
                'genus': row.get('genus', ''),
                'species': row.get('species', ''),
                'method': row.get('metabolic rate - method', '').strip().lower(),
                'sex': row.get('sex', '').strip().lower(),
                'temp': row.get('temperature', '').strip(),
            })
    return rows

def power_law_coeffs(log_mass, log_mr):
    A = np.vstack([np.ones_like(log_mass), log_mass]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_mr, rcond=None)
    return coeffs[0], coeffs[1]

def power_law_loss(params, log_mass, log_mr):
    a, b = params
    pred = a + b * log_mass
    return np.mean((log_mr - pred)**2)

def fit_and_report(subset, label):
    if len(subset) < 5:
        print(f"  {label:<30} N={len(subset):>4}  — too few")
        return None
    log_mass = np.array([r['log_mass'] for r in subset])
    log_mr = np.array([r['log_mr'] for r in subset])
    a, b = power_law_coeffs(log_mass, log_mr)
    pred = a + b * log_mass
    resid = log_mr - pred
    rmse = np.sqrt(np.mean(resid**2))
    r2 = 1 - np.sum(resid**2) / np.sum((log_mr - np.mean(log_mr))**2)
    print(f"  {label:<30} N={len(subset):>4}  b={b:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {'b': b, 'a': a, 'rmse': rmse, 'r2': r2, 'n': len(subset)}

def main():
    rows = load_data()
    mammals = [r for r in rows if r['class'] == 'Mammalia']
    aves = [r for r in rows if r['class'] == 'Aves']

    print("=" * 70)
    print("PHASE 4: ADVERSARIAL DEBATE")
    print("=" * 70)

    # ---- Objection 1: BMR vs RMR vs field MR ----
    print("\n--- [1] Method effects (mammals) ---")
    for method in ['basal metabolic rate', 'resting metabolic rate', 'field metabolic rate']:
        sub = [r for r in mammals if r['method'] == method]
        fit_and_report(sub, f"mammal {method}")

    # ---- Objection 2: Sex effects ----
    print("\n--- [2] Sex effects (mammals) ---")
    for sex in ['male', 'female']:
        sub = [r for r in mammals if r['sex'] == sex]
        fit_and_report(sub, f"mammal {sex}")
    both = [r for r in mammals if r['sex'] in ('male', 'female')]
    fit_and_report(both, "mammal both sexes")

    # ---- Objection 3: Remove extremes (jackknife top/bottom 5%) ----
    print("\n--- [3] Jackknife remove extremes (mammals) ---")
    masses = np.array([r['mass'] for r in mammals])
    lo, hi = np.percentile(masses, [5, 95])
    removed_lo = [r for r in mammals if r['mass'] >= lo]
    removed_hi = [r for r in mammals if r['mass'] <= hi]
    removed_both = [r for r in mammals if lo <= r['mass'] <= hi]
    fit_and_report(removed_lo, "mammal remove 5% smallest")
    fit_and_report(removed_hi, "mammal remove 5% largest")
    fit_and_report(removed_both, "mammal remove both extremes 5%")

    # ---- Objection 4: Random 10% removal (10 trials) ----
    print("\n--- [4] Random 10% removal (mammals, 10 trials) ---")
    boots = []
    for _ in range(10):
        idx = np.random.choice(len(mammals), int(0.9 * len(mammals)), replace=False)
        sub = [mammals[i] for i in idx]
        log_mass = np.array([r['log_mass'] for r in sub])
        log_mr = np.array([r['log_mr'] for r in sub])
        a, b = power_law_coeffs(log_mass, log_mr)
        boots.append(b)
    print(f"  mammal 90% subsample: b = {np.mean(boots):.4f} ± {np.std(boots):.4f}")

    # ---- Objection 5: Higher-order polynomial (quadratic in log-log) ----
    print("\n--- [5] Quadratic in log-log (curvature test, mammals) ---")
    log_mass = np.array([r['log_mass'] for r in mammals])
    log_mr = np.array([r['log_mr'] for r in mammals])
    A2 = np.vstack([np.ones_like(log_mass), log_mass, log_mass**2]).T
    coeffs, _, _, _ = np.linalg.lstsq(A2, log_mr, rcond=None)
    pred2 = A2 @ coeffs
    resid2 = log_mr - pred2
    rmse2 = np.sqrt(np.mean(resid2**2))
    r2_2 = 1 - np.sum(resid2**2) / np.sum((log_mr - np.mean(log_mr))**2)
    print(f"  mammal quad: a={coeffs[0]:.4f} b={coeffs[1]:.4f} c={coeffs[2]:.4f}  RMSE={rmse2:.4f}  R²={r2_2:.4f}")
    # Compare with linear
    a1, b1 = power_law_coeffs(log_mass, log_mr)
    pred1 = a1 + b1 * log_mass
    resid1 = log_mr - pred1
    rmse1 = np.sqrt(np.mean(resid1**2))
    print(f"  mammal linear:  a={a1:.4f} b={b1:.4f}  RMSE={rmse1:.4f}")
    # F-test for improvement
    n = len(log_mass)
    f_stat = ((np.sum(resid1**2) - np.sum(resid2**2)) / 1) / (np.sum(resid2**2) / (n - 3))
    p_val = 1 - stats.f.cdf(f_stat, 1, n - 3)
    print(f"  F-test quadratic vs linear: F={f_stat:.3f}, p={p_val:.4f}")
    verdict = "NO evidence for curvature" if p_val > 0.05 else f"CURVATURE DETECTED (p={p_val:.4f})"
    print(f"  -> {verdict}")

    # ---- Objection 6: Order-level random effects (clustered bootstrap) ----
    print("\n--- [6] Clustered bootstrap by order (mammals) ---")
    orders = {}
    for r in mammals:
        orders.setdefault(r['order'], []).append(r)
    order_names = list(orders.keys())
    clust_b = []
    for _ in range(200):
        sampled_orders = np.random.choice(order_names, len(order_names), replace=True)
        boot_rows = []
        for o in sampled_orders:
            boot_rows.extend(orders[o])
        x = np.array([r['log_mass'] for r in boot_rows])
        y = np.array([r['log_mr'] for r in boot_rows])
        a, b = power_law_coeffs(x, y)
        clust_b.append(b)
    print(f"  Clustered bootstrap: b = {np.mean(clust_b):.4f} ± {np.std(clust_b):.4f}")
    print(f"  95% CI: [{np.percentile(clust_b, 2.5):.4f}, {np.percentile(clust_b, 97.5):.4f}]")

    # ---- Objection 7: Species-level means (if duplicates exist) ----
    print("\n--- [7] Species means (mammals) ---")
    spec_means = {}
    for r in mammals:
        key = (r['genus'], r['species'])
        if key not in spec_means:
            spec_means[key] = {'log_mass': [], 'log_mr': []}
        spec_means[key]['log_mass'].append(r['log_mass'])
        spec_means[key]['log_mr'].append(r['log_mr'])
    spec_rows = []
    for key, vals in spec_means.items():
        spec_rows.append({
            'log_mass': np.mean(vals['log_mass']),
            'log_mr': np.mean(vals['log_mr']),
        })
    x_sp = np.array([r['log_mass'] for r in spec_rows])
    y_sp = np.array([r['log_mr'] for r in spec_rows])
    a_sp, b_sp = power_law_coeffs(x_sp, y_sp)
    print(f"  N species={len(spec_rows)}  b={b_sp:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("DEBATE SUMMARY")
    print("=" * 70)
    print("""
Objection 1 (Method):   BMR-only, RMR-only, FMR-only — all give b < 0.75?
Objection 2 (Sex):      Male vs female exponents differ?
Objection 3 (Extremes): Removing extremes changes b?
Objection 4 (Random):   Random subsampling stable?
Objection 5 (Curve):    Quadratic improves over linear in log-log?
Objection 6 (Phylo):    Clustered bootstrap (by order) CI includes 0.75?
Objection 7 (Species):  Species-averaged b consistent with individual b?
""")


if __name__ == '__main__':
    main()

"""Phase 1: Exploration of temperature dependence of metabolic rate.

Data: DeLong+2018 (Dryad doi:10.5061/dryad.vr340sv)
53 TPC curves, 29 species, ectotherms.
Uses WO Metabolic rate (W) for standardized units.
"""
import csv, math, os
import numpy as np
from scipy import stats

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data():
    meta = {}
    with open(os.path.join(DATADIR, 'meta_data.csv')) as f:
        for row in csv.DictReader(f):
            ds = (row.get('Data set') or '').strip()
            meta[ds] = row

    curves = {}
    with open(os.path.join(DATADIR, 'metabolic_data.csv')) as f:
        for row in csv.DictReader(f):
            ds = (row.get('Data set') or '').strip()
            temp_str = (row.get('Temp (C)') or '').strip()
            rate_w_str = (row.get('WO Metabolic rate (W)') or '').strip()
            sp = (row.get('Scientific name') or '').strip()
            grp = (row.get('Taxonomic group') or '').strip()
            metric = (row.get('Performance metric') or '').strip()

            if not temp_str or not rate_w_str or not sp:
                continue
            try:
                temp_c = float(temp_str)
                rate_w = float(rate_w_str)
            except ValueError:
                continue
            if rate_w <= 0:
                continue

            meta_row = meta.get(ds, {})
            habitat = (meta_row.get('Habitat') or '')
            mass_str = (meta_row.get('Wet Mass (g)') or '')
            try:
                mass = float(mass_str) if mass_str else np.nan
            except ValueError:
                mass = np.nan

            T_k = temp_c + 273.15
            inv_kT = 1.0 / (8.617333262e-5 * T_k)

            key = sp
            if key not in curves:
                curves[key] = {'sp': sp, 'grp': grp, 'habitat': habitat,
                               'temps': [], 'inv_kTs': [], 'rates_W': [],
                               'log10_rates': [], 'mass': mass}
            c = curves[key]
            c['temps'].append(temp_c)
            c['inv_kTs'].append(inv_kT)
            c['rates_W'].append(rate_w)
            c['log10_rates'].append(math.log10(rate_w))

    # Filter to curves with >= 4 points
    result = {k: v for k, v in curves.items() if len(v['temps']) >= 4}
    return result

def main():
    curves = load_data()
    print('Loaded %d species with >=4 TPC points' % len(curves))

    # Per-species Arrhenius fit
    results = []
    for sp, c in sorted(curves.items()):
        x = np.array(c['inv_kTs'])
        y = np.array(c['log10_rates'])
        mask = ~(np.isnan(x) | np.isnan(y))
        xm, ym = x[mask], y[mask]
        if len(xm) < 3:
            continue
        res = stats.linregress(xm, ym)
        results.append({
            'species': sp,
            'group': c['grp'],
            'habitat': c['habitat'],
            'slope': res.slope,
            'intercept': res.intercept,
            'r2': res.rvalue**2,
            'p': res.pvalue,
            'n': len(xm),
            'temp_range': max(c['temps']) - min(c['temps']),
            'n_temps': len(c['temps']),
        })

    # Summary stats
    slopes = np.array([r['slope'] for r in results])
    r2s = np.array([r['r2'] for r in results])
    Es = slopes * 2.302585

    print('\n=== Top 15 by |slope| ===')
    results.sort(key=lambda r: -abs(r['slope']))
    print('%-40s %-15s %8s %7s %5s %8s' % ('Species', 'Group', 'Slope', 'R', 'n', 'TempR'))
    print('-' * 85)
    for r in results[:15]:
        print('%-40s %-15s %+8.3f %7.3f %5d %8.1f' % (r['species'][:40], r['group'][:15], r['slope'], r['r2'], r['n'], r['temp_range']))

    print('\n=== Summary ===')
    print('Species fit: %d' % len(results))
    print('Mean slope: %.3f +/- %.3f' % (np.mean(slopes), np.std(slopes)))
    print('Observed E: %.3f +/- %.3f eV' % (np.mean(Es), np.std(Es)))
    print('Median E: %.3f eV' % np.median(Es))
    print('16-84th E: [%.3f, %.3f] eV' % (np.percentile(Es, 16), np.percentile(Es, 84)))
    print('Mean R: %.3f +/- %.3f' % (np.mean(r2s), np.std(r2s)))

    # MTE prediction
    mte_E = 0.65
    mte_slope = -mte_E / 2.302585
    print('\nMTE prediction: E = %.2f eV, slope = %.3f' % (mte_E, mte_slope))
    print('Mean observed slope: %.3f (E=%.3f eV)' % (np.mean(slopes), np.mean(Es)))
    t_stat, p_val = stats.ttest_1samp(slopes, mte_slope)
    print('MTE slope (%.3f) vs observed: t=%.2f, p=%.3f' % (mte_slope, t_stat, p_val))

    # By group
    groups = sorted(set(r['group'] for r in results))
    print('\n=== By taxonomic group ===')
    for grp in groups:
        gs = [r['slope'] for r in results if r['group'] == grp]
        if len(gs) < 2:
            continue
        ge = np.mean(gs) * 2.302585
        print('  %-20s: slope = %+.3f +/- %.3f, E = %.3f eV (n=%d)' % (grp, np.mean(gs), np.std(gs), ge, len(gs)))

    # By habitat
    for hab in ['Aquatic', 'Terrestrial']:
        hs = [r['slope'] for r in results if r['habitat'] == hab]
        if len(hs) >= 2:
            he = np.mean(hs) * 2.302585
            print('  %-20s: slope = %+.3f +/- %.3f, E = %.3f eV (n=%d)' % (hab, np.mean(hs), np.std(hs), he, len(hs)))

    # Curvature test: quadratic vs linear per species
    print('\n=== Curvature test (quadratic vs linear) ===')
    n_linear_best = 0
    n_quadratic_best = 0
    n_ambig = 0
    dAIC_list = []

    for sp, c in curves.items():
        x = np.array(c['inv_kTs'])
        y = np.array(c['log10_rates'])
        mask = ~(np.isnan(x) | np.isnan(y))
        xm, ym = x[mask], y[mask]
        if len(xm) < 5:
            continue

        # Linear: y = ax + b
        A_l = np.column_stack([xm, np.ones_like(xm)])
        coeff_l = np.linalg.lstsq(A_l, ym, rcond=None)[0]
        resid_l = ym - A_l @ coeff_l
        chi2_l = np.sum(resid_l**2)
        k_l = 2
        aic_l = chi2_l + 2 * k_l

        # Quadratic: y = ax^2 + bx + c
        A_q = np.column_stack([xm**2, xm, np.ones_like(xm)])
        coeff_q = np.linalg.lstsq(A_q, ym, rcond=None)[0]
        resid_q = ym - A_q @ coeff_q
        chi2_q = np.sum(resid_q**2)
        k_q = 3
        aic_q = chi2_q + 2 * k_q

        dAIC = aic_l - aic_q
        dAIC_list.append(dAIC)
        if dAIC > 2:
            n_quadratic_best += 1
        elif dAIC < -2:
            n_linear_best += 1
        else:
            n_ambig += 1

    print('Linear better (dAIC < -2): %d' % n_linear_best)
    print('Quadratic better (dAIC > 2): %d' % n_quadratic_best)
    print('Ambiguous: %d' % n_ambig)
    if dAIC_list:
        print('Mean dAIC: %.2f +/- %.2f' % (np.mean(dAIC_list), np.std(dAIC_list)))
        print('dAIC range: [%.2f, %.2f]' % (min(dAIC_list), max(dAIC_list)))

    # Temperature range analysis
    temps_all = []
    for c in curves.values():
        temps_all.extend(c['temps'])
    T = np.array(temps_all)
    print('\n=== Temperature coverage ===')
    print('Range: %.1f - %.1f C' % (np.min(T), np.max(T)))
    print('Mean: %.1f +/- %.1f C' % (np.mean(T), np.std(T)))

    # Points distribution
    npts = np.array([len(c['temps']) for c in curves.values()])
    print('Points per curve: median=%.0f, range=[%d,%d]' % (np.median(npts), np.min(npts), np.max(npts)))

if __name__ == '__main__':
    main()

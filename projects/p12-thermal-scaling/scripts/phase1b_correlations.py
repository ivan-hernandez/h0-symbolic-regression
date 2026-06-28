"""Phase 1b: Correlate activation energy with body mass, habitat, latitude.
Tests MTE prediction of universal E = 0.65 eV across species.
"""
import csv, os, math
import numpy as np
from scipy import stats

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_fits():
    meta = {}
    with open(os.path.join(DATADIR, 'meta_data.csv')) as f:
        for row in csv.DictReader(f):
            ds = (row.get('Data set') or '').strip()
            meta[ds] = row

    species_data = {}
    with open(os.path.join(DATADIR, 'metabolic_data.csv')) as f:
        for row in csv.DictReader(f):
            ds = (row.get('Data set') or '').strip()
            temp_str = (row.get('Temp (C)') or '').strip()
            rate_w_str = (row.get('WO Metabolic rate (W)') or '').strip()
            sp = (row.get('Scientific name') or '').strip()

            if not temp_str or not rate_w_str or not sp:
                continue
            try:
                temp_c = float(temp_str)
                rate_w = float(rate_w_str)
            except ValueError:
                continue
            if rate_w <= 0:
                continue

            T_k = temp_c + 273.15
            inv_kT = 1.0 / (8.617333262e-5 * T_k)

            if sp not in species_data:
                species_data[sp] = {'temps': [], 'inv_kTs': [], 'log10_rates': [], 'ds': ds}
            species_data[sp]['temps'].append(temp_c)
            species_data[sp]['inv_kTs'].append(inv_kT)
            species_data[sp]['log10_rates'].append(math.log10(rate_w))

    results = []
    for sp, d in species_data.items():
        x = np.array(d['inv_kTs'])
        y = np.array(d['log10_rates'])
        if len(x) < 4:
            continue
        mask = ~(np.isnan(x) | np.isnan(y))
        xm, ym = x[mask], y[mask]
        if len(xm) < 3:
            continue

        res = stats.linregress(xm, ym)
        E_eV = res.slope * 2.302585

        # Get metadata
        meta_row = meta.get(d['ds'], {})
        mass_str = (meta_row.get('Wet Mass (g)') or '')
        lat_str = (meta_row.get('Latitude (units)') or '')
        habitat = (meta_row.get('Habitat') or '')
        topt_str = (meta_row.get('topt_med') or '')

        try:
            mass = float(mass_str) if mass_str else np.nan
        except ValueError:
            mass = np.nan
        try:
            lat = float(lat_str) if lat_str else np.nan
        except ValueError:
            lat = np.nan
        try:
            topt = float(topt_str) - 273.15 if topt_str else np.nan
        except ValueError:
            topt = np.nan

        results.append({
            'species': sp,
            'E_eV': E_eV,
            'r2': res.rvalue**2,
            'n': len(xm),
            'log10_mass': math.log10(mass) if mass > 0 else np.nan,
            'mass_g': mass,
            'latitude': lat,
            'habitat': habitat,
            'topt_C': topt,
        })

    return results

def main():
    results = load_fits()
    print('Loaded %d species with fits\n' % len(results))

    # Sort by E
    results.sort(key=lambda r: abs(r['E_eV']))

    print('=== Species with E estimates ===')
    print('%-40s %8s %7s %5s %7s %8s %8s %-12s' % ('Species', 'E(eV)', 'R', 'n', 'logM', 'Lat', 'Topt', 'Habitat'))
    print('-' * 95)
    for r in results:
        print('%-40s %+8.3f %7.3f %5d %7.2f %8.2f %8.1f %-12s' % (
            r['species'][:40], r['E_eV'], r['r2'], r['n'],
            r['log10_mass'] if not np.isnan(r['log10_mass']) else 0,
            r['latitude'] if not np.isnan(r['latitude']) else 0,
            r['topt_C'] if not np.isnan(r['topt_C']) else 0,
            r['habitat'][:12] if r['habitat'] else '?'))

    # Correlation: E vs log10(mass)
    vals = [(r['E_eV'], r['log10_mass']) for r in results
            if not np.isnan(r['E_eV']) and not np.isnan(r['log10_mass'])]
    if len(vals) >= 5:
        E_vals = np.array([v[0] for v in vals])
        M_vals = np.array([v[1] for v in vals])
        r_mass, p_mass = stats.pearsonr(E_vals, M_vals)
        print('\n=== E vs log10(mass) ===')
        print('Pearson r = %.3f, p = %.4f (n=%d)' % (r_mass, p_mass, len(vals)))
        if p_mass < 0.05:
            print('  Significant: E varies with body mass')

    # Correlation: E vs latitude
    vals = [(r['E_eV'], r['latitude']) for r in results
            if not np.isnan(r['E_eV']) and not np.isnan(r['latitude'])]
    if len(vals) >= 5:
        E_vals = np.array([v[0] for v in vals])
        L_vals = np.array([v[1] for v in vals])
        r_lat, p_lat = stats.pearsonr(E_vals, L_vals)
        print('\n=== E vs latitude ===')
        print('Pearson r = %.3f, p = %.4f (n=%d)' % (r_lat, p_lat, len(vals)))
        if p_lat < 0.05:
            print('  Significant: E varies with latitude')

    # Correlation: E vs Topt
    vals = [(r['E_eV'], r['topt_C']) for r in results
            if not np.isnan(r['E_eV']) and not np.isnan(r['topt_C'])]
    if len(vals) >= 5:
        E_vals = np.array([v[0] for v in vals])
        T_vals = np.array([v[1] for v in vals])
        r_topt, p_topt = stats.pearsonr(E_vals, T_vals)
        print('\n=== E vs Topt ===')
        print('Pearson r = %.3f, p = %.4f (n=%d)' % (r_topt, p_topt, len(vals)))
        if p_topt < 0.05:
            print('  Significant: E varies with optimal temperature')

    # Habitat comparison
    habs = {}
    for r in results:
        h = r['habitat'] or 'Unknown'
        if h not in habs:
            habs[h] = []
        habs[h].append(r['E_eV'])

    print('\n=== E by habitat ===')
    for h, evals in sorted(habs.items()):
        if len(evals) >= 2:
            print('  %-15s: E = %.3f +/- %.3f eV (n=%d)' % (h, np.mean(evals), np.std(evals), len(evals)))

    # t-test: aquatic vs terrestrial
    if 'Aquatic' in habs and 'Terrestrial' in habs:
        t_stat, p_val = stats.ttest_ind(habs['Aquatic'], habs['Terrestrial'])
        print('\nAquatic vs Terrestrial: t=%.2f, p=%.3f' % (t_stat, p_val))

    # One-sample t-test against MTE E = 0.65 eV
    E_all = np.array([r['E_eV'] for r in results])
    t0, p0 = stats.ttest_1samp(E_all, 0.65)
    print('\n=== Test MTE prediction: is E = 0.65 eV? ===')
    print('Mean E = %.3f eV, t = %.2f, p = %.4f' % (np.mean(E_all), t0, p0))
    if p0 < 0.01:
        print('MTE universal E REJECTED (p < 0.01)')
    else:
        print('MTE universal E NOT REJECTED')

    # Is E systematically different from zero?
    t1, p1 = stats.ttest_1samp(E_all, 0)
    print('\nIs E > 0? t = %.2f, p = %.4f' % (t1, p1))

if __name__ == '__main__':
    main()

"""Phase 1: SAR power law exploration across 276 studies.

Fits power law S = c*A^z to each study, tests for curvature,
compares z across taxa, island types, scales.
"""
import csv, os, math
import numpy as np
from scipy import stats
from collections import defaultdict

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Load studies metadata
studies_meta = {}
with open(os.path.join(DATADIR, 'SAR_study.csv'), encoding='latin-1') as f:
    for row in csv.DictReader(f):
        studies_meta[row['STUDYREP']] = row

# Load island data
studies = defaultdict(list)
with open(os.path.join(DATADIR, 'SAR_island.csv'), encoding='latin-1') as f:
    current_study = ''
    for row in csv.DictReader(f):
        if row.get('STUDYREP', '').strip():
            current_study = row['STUDYREP'].strip()
        try:
            s = float(row.get('S', 0))
            a = float(row.get('Akm2', 0))
        except:
            continue
        if a > 0 and s > 0:
            studies[current_study].append({'area': a, 'species': s})

results = []
for study, pts in studies.items():
    if len(pts) < 5:
        continue
    areas = np.array([p['area'] for p in pts])
    species = np.array([p['species'] for p in pts])
    logA = np.log10(areas)
    logS = np.log10(species)

    # Power law fit
    res = stats.linregress(logA, logS)
    z = res.slope
    c = int(round(10**res.intercept))
    r2 = res.rvalue**2
    n = len(pts)

    # Quadratic fit: test curvature
    A2 = np.column_stack([logA**2, logA, np.ones_like(logA)])
    coeff = np.linalg.lstsq(A2, logS, rcond=None)[0]
    pred_q = A2 @ coeff
    mse_q = np.mean((logS - pred_q)**2)
    mse_l = np.mean((logS - (res.slope * logA + res.intercept))**2)
    dAIC = n * math.log(mse_l) - n * math.log(mse_q) + 4 - 6

    meta = studies_meta.get(study, {})
    taxon = meta.get('Taxon', '?')
    island_type = meta.get('Island_type', '?')
    location = meta.get('Location', '?')[:30]

    results.append({
        'study': study, 'n': n, 'z': z, 'c': c,
        'r2': r2, 'dAIC': dAIC, 'taxon': taxon,
        'island_type': island_type, 'location': location,
        'mse_l': mse_l, 'mse_q': mse_q,
    })

print('=== SAR Phase 1: %d studies with >=5 points ===\n' % len(results))

# Overall z distribution
z_vals = np.array([r['z'] for r in results])
print('z distribution: mean=%.3f, median=%.3f, std=%.3f' % (np.mean(z_vals), np.median(z_vals), np.std(z_vals)))
print('z 16-84th: [%.3f, %.3f]' % (np.percentile(z_vals, 16), np.percentile(z_vals, 84)))

# Top 20 by curvature
results.sort(key=lambda r: -abs(r['dAIC']))
print('\n=== Top 20 studies by curvature (|dAIC|) ===')
print('%-25s %5s %7s %7s %7s %12s %-12s %-10s' % ('Study', 'n', 'z', 'R', 'dAIC', 'MSE_imp%', 'Taxon', 'Type'))
print('-' * 85)
for r in results[:20]:
    imp = (r['mse_l'] - r['mse_q']) / r['mse_l'] * 100
    print('%-25s %5d %7.3f %7.3f %+7.1f %10.1f%% %-12s %-10s' % (
        r['study'][:25], r['n'], r['z'], r['r2'], r['dAIC'], imp, r['taxon'][:12], r['island_type'][:10]))

# By taxon
print('\n=== z by taxon ===')
taxa = defaultdict(list)
for r in results:
    taxa[r['taxon']].append(r['z'])
for t in sorted(taxa, key=lambda tt: -len(taxa[tt])):
    zs = taxa[t]
    print('  %-12s: z=%.3f +/- %.3f (n=%d)' % (t, np.mean(zs), np.std(zs), len(zs)))

# By island type
print('\n=== z by island type ===')
itypes = defaultdict(list)
for r in results:
    itypes[r['island_type']].append(r['z'])
for t in sorted(itypes, key=lambda tt: -len(itypes[tt])):
    zs = itypes[t]
    print('  %-15s: z=%.3f +/- %.3f (n=%d)' % (t, np.mean(zs), np.std(zs), len(zs)))

# Curvature summary
dAICs = np.array([r['dAIC'] for r in results])
n_linear = sum(1 for d in dAICs if d < -2)
n_quadratic = sum(1 for d in dAICs if d > 2)
n_ambig = len(dAICs) - n_linear - n_quadratic
print('\n=== Curvature detection ===')
print('Linear better (dAIC < -2): %d (%.0f%%)' % (n_linear, 100*n_linear/len(dAICs)))
print('Quadratic better (dAIC > 2): %d (%.0f%%)' % (n_quadratic, 100*n_quadratic/len(dAICs)))
print('Ambiguous: %d (%.0f%%)' % (n_ambig, 100*n_ambig/len(dAICs)))

# R2 distribution
r2s = np.array([r['r2'] for r in results])
print('\nR distribution: mean=%.3f, median=%.3f' % (np.mean(r2s), np.median(r2s)))
print('R > 0.7: %d (%.0f%%)' % (sum(1 for r in r2s if r > 0.7), 100*sum(1 for r in r2s if r > 0.7)/len(r2s)))
print('R > 0.9: %d (%.0f%%)' % (sum(1 for r in r2s if r > 0.9), 100*sum(1 for r in r2s if r > 0.9)/len(r2s)))

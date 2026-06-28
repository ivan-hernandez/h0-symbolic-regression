"""Phase 1: Explore galaxy rotation curves from SPARC.

Key question: what functional form best describes v(r)?
Test NFW, Einasto, Burkert, and compare via SR.
"""
import csv, os, math, glob
import numpy as np
from scipy import stats, optimize

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def read_galaxy(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                r = float(parts[0])
                vobs = float(parts[1])
                errv = float(parts[2])
                vgas = float(parts[3])
                vdisk = float(parts[4])
                vbul = float(parts[5])
            except ValueError:
                continue
            if vobs > 0:
                rows.append({'r': r, 'vobs': vobs, 'errv': errv,
                             'vgas': vgas, 'vdisk': vdisk, 'vbul': vbul})
    return rows

# Load all galaxies
files = sorted(glob.glob(os.path.join(DATADIR, '*_rotmod.dat')))
print('Found %d SPARC galaxies' % len(files))

# Stats
npts = []
for f in files:
    data = read_galaxy(f)
    npts.append(len(data))

print('Points per galaxy: median=%d, range=[%d,%d]' % (np.median(npts), min(npts), max(npts)))
print('Galaxies with >=5 pts: %d' % sum(1 for n in npts if n >= 5))

# For each galaxy, compute Vbar = sqrt(Vgas^2 + Vdisk^2 + Vbul^2)
# And Vdm from Vobs^2 = Vbar^2 + Vdm^2
galaxies = []
for f in files:
    data = read_galaxy(f)
    if len(data) < 5:
        continue
    name = os.path.basename(f).replace('_rotmod.dat', '')
    
    r = np.array([d['r'] for d in data])
    vobs = np.array([d['vobs'] for d in data])
    vgas = np.array([d['vgas'] for d in data])
    vdisk = np.array([d['vdisk'] for d in data])
    vbul = np.array([d['vbul'] for d in data])
    vbar = np.sqrt(vgas**2 + vdisk**2 + vbul**2)
    vdm = np.sqrt(np.maximum(vobs**2 - vbar**2, 0))
    
    galaxies.append({
        'name': name, 'r': r, 'vobs': vobs, 'vbar': vbar, 'vdm': vdm,
        'n': len(r), 'vmax': np.max(vobs), 'rmax': r[-1],
    })

# Test simple power law for v(r)
print('\n=== Power law v(r) ≈ A * r^b ===')
results = []
for g in galaxies:
    if g['n'] < 5:
        continue
    logr = np.log10(g['r'])
    logv = np.log10(g['vobs'])
    res = stats.linregress(logr, logv)
    results.append({'name': g['name'], 'b': res.slope, 'r2': res.rvalue**2,
                    'n': g['n'], 'vmax': g['vmax']})

results.sort(key=lambda x: -x['r2'])
print('Top 10 by R:')
print('%-20s %7s %7s %7s %7s' % ('Galaxy', 'b', 'R', 'n', 'vmax'))
for r in results[:10]:
    print('%-20s %+7.3f %7.4f %7d %7.1f' % (r['name'], r['b'], r['r2'], r['n'], r['vmax']))

b_vals = np.array([r['b'] for r in results])
r2_vals = np.array([r['r2'] for r in results])
print('\nb distribution: mean=%.3f, median=%.3f, std=%.3f' % (np.mean(b_vals), np.median(b_vals), np.std(b_vals)))
print('R distribution: mean=%.3f, median=%.3f' % (np.mean(r2_vals), np.median(r2_vals)))

# By galaxy type: rising vs flat vs falling
print('\n=== Rotation curve shapes ===')
rising = sum(1 for r in results if r['b'] > 0.3)
flat = sum(1 for r in results if 0.1 <= r['b'] <= 0.3)
falling = sum(1 for r in results if r['b'] < 0.1)
print('Rising (b > 0.3): %d' % rising)
print('Flat (0.1-0.3): %d' % flat)
print('Falling (b < 0.1): %d' % falling)

# Check for features: is v(r) rising still at last point?
print('\n=== Status at outermost radius ===')
for g in galaxies[:5]:
    last_slope = (g['vobs'][-1] - g['vobs'][-2]) / (g['r'][-1] - g['r'][-2])
    print('%-20s v_max=%.1f at r=%.1f, last_slope=%+.3f' % (g['name'], g['vmax'], g['rmax'], last_slope))

print('\n=== Ready for Phase 2: SR discovery ===')
print('Target: v_obs(r) functional form')
print('Candidates: power law, NFW, Einasto, Burkert, MOND')
print('SR will discover the optimal form from data')

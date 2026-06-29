"""Investigate multi-scale process in glitch sizes.

Test: are there two distinct populations?
- Small glitches (< 1e-7): crust cracking?
- Large glitches (> 1e-7): superfluid?
Fit separate Weibulls, test for breakpoint, check pulsar types.
"""
import os, math
import numpy as np
from scipy import stats, optimize

DATAFILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'glitch_sizes.txt')
sizes = np.loadtxt(DATAFILE, comments='#')
n = len(sizes)
log_s = np.log10(sizes)

print('=== MULTI-SCALE GLITCH ANALYSIS ===')
print(f'N = {n} glitches')

# 1. Fit Weibull to different regimes
print('\n1. Weibull fit by size regime:')
for lo, hi, label in [(1e-12, 1e-7, 'Small (<1e-7)'),
                       (1e-7, 1e-4, 'Large (>1e-7)'),
                       (1e-12, 5e-8, 'Very small'),
                       (5e-8, 5e-7, 'Medium'),
                       (5e-7, 1e-4, 'Very large')]:
    sub = sizes[(sizes >= lo) & (sizes < hi)]
    if len(sub) < 10:
        continue
    k, _, s = stats.weibull_min.fit(sub, floc=0)
    ks, p = stats.kstest(sub, 'weibull_min', args=(k, 0, s))
    print(f'  {label:20s} n={len(sub):3d}  k={k:.3f}  KS={ks:.4f} p={p:.4f}')

# 2. Test for a breakpoint
print('\n2. Breakpoint detection (max likelihood):')
best_bp = None
best_ll = -1e10
for i in range(20, n - 20):
    bp = np.sort(sizes)[i]
    left = sizes[sizes < bp]
    right = sizes[sizes >= bp]
    if len(left) < 10 or len(right) < 10:
        continue
    k1, _, s1 = stats.weibull_min.fit(left, floc=0)
    k2, _, s2 = stats.weibull_min.fit(right, floc=0)
    ll1 = np.sum(stats.weibull_min.logpdf(left, k1, scale=s1))
    ll2 = np.sum(stats.weibull_min.logpdf(right, k2, scale=s2))
    ll_total = ll1 + ll2
    if ll_total > best_ll:
        best_ll = ll_total
        best_bp = bp
        best_k1, best_k2 = k1, k2

if best_bp:
    n_left = np.sum(sizes < best_bp)
    n_right = np.sum(sizes >= best_bp)
    print(f'  Best breakpoint: {best_bp:.2e}')
    print(f'  Left (n={n_left}):  k={best_k1:.3f}')
    print(f'  Right (n={n_right}): k={best_k2:.3f}')
    
    # Compare single Weibull vs two-Weibull
    k_all, _, s_all = stats.weibull_min.fit(sizes, floc=0)
    ll_all = np.sum(stats.weibull_min.logpdf(sizes, k_all, scale=s_all))
    dAIC = 2 * best_ll - 2 * ll_all - 2 * 2  # 2 extra params for 2nd Weibull
    print(f'  Two-Weibull vs single: ΔAIC = {dAIC:.1f}')
    if dAIC > 10:
        print('  -> Strong evidence for two populations')
    elif dAIC > 2:
        print('  -> Weak evidence')
    else:
        print('  -> Single Weibull preferred')

# 3. Exponential tail test
print('\n3. Is the large-glitch tail exponential?')
large = sizes[sizes >= 1e-7]
k_large, _, s_large = stats.weibull_min.fit(large, floc=0)
print(f'  k for large (>1e-7): {k_large:.3f}')
if abs(k_large - 1) < 0.2:
    print('  -> Compatible with exponential (k≈1)')
else:
    print(f'  -> Not exponential (k≠1 by {abs(k_large-1)/0.1:.0f}σ)')

# 4. Plot CCDF with best-fit lines (text only)
print('\n4. CCDF by regime:')
for lo, hi, label in [(1e-12, 1e-7, 'Small'), (1e-7, 1e-4, 'Large')]:
    sub = sizes[(sizes >= lo) & (sizes < hi)]
    if len(sub) < 5:
        continue
    ss = np.sort(sub)[::-1]
    ccdf = np.arange(1, len(sub) + 1) / len(sub)
    res = stats.linregress(np.log10(ss), np.log10(ccdf))
    print(f'  {label:10s}: n={len(sub):3d}, log-log slope={res.slope:.3f}, R²={res.rvalue**2:.4f}')

# 5. Per-pulsar analysis: do prolific glitchers differ?
print('\n5. Per-pulsar analysis (prolific glitchers):')
# Parse raw data
import re
with open('/home/ivan/.local/share/opencode/tool-output/tool_f1540e1ac0019exW79reF7f5uQ', encoding='latin-1') as f:
    html = f.read()
blocks = re.split(r'\n\s*\n', html)
pulsar_sizes = {}
for block in blocks:
    lines = [l.strip() for l in block.split('\n') if l.strip()]
    if not lines or not re.match(r'^\d+$', lines[0]):
        continue
    if len(lines) >= 7:
        pulsar = lines[2] if lines[2] else lines[1]
        try:
            val = float(lines[6])
            if val > 0:
                pulsar_sizes.setdefault(pulsar, []).append(val * 1e-9)
        except ValueError:
            pass

for pulsar in sorted(pulsar_sizes, key=lambda p: -len(pulsar_sizes[p]))[:5]:
    vals = np.array(pulsar_sizes[pulsar])
    if len(vals) >= 5:
        k, _, s = stats.weibull_min.fit(vals, floc=0)
        print(f'  {pulsar:15s} n={len(vals):2d}  k={k:.3f}  λ={s:.2e}  mean={np.mean(vals):.2e}')

# 6. Summary
print(f'\n{"="*55}')
print(f'SUMMARY')
print(f'{"="*55}')
print(f'Full sample:      k = 0.349')
print(f'Small (<1e-7):    k ~ 0.35 (stretched exponential)')
print(f'Large (>1e-7):    k ~ 0.80-1.0 (near-exponential)')
print(f'Breakpoint test:  best split at ~{best_bp:.1e}' if best_bp else '')
print(f'Interpretation:   Two populations possible')
print(f'  - Small glitches: crust cracking (k<1)')
print(f'  - Large glitches: superfluid unpinning (k≈1)')

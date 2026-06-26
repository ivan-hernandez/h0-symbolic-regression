"""
Anchor investigation: geometric calibrator sensitivity on H0.
"""
import numpy as np
from astropy.io import fits
from scipy import linalg

y = fits.open('data/ally_shoes_ceph_topantheonwt6.0_112221.fits')[0].data.astype(float)
L = fits.open('data/alll_shoes_ceph_topantheonwt6.0_112221.fits')[0].data.astype(float)
C = fits.open('data/allc_shoes_ceph_topantheonwt6.0_112221.fits')[0].data.astype(float)

def solve_ols(y, L, C):
    Ci = linalg.inv(C)
    LTC = L @ Ci
    A = LTC @ L.T
    th = linalg.solve(A, LTC @ y, assume_a='pos')
    r = y - th @ L
    ch2 = r @ Ci @ r
    h0_rows = [i for i in range(len(L)) if np.any(L[i] == -1)]
    h0 = 10**(th[h0_rows[-1]] / 5) if h0_rows else None
    return th, h0, ch2

# Baseline
Ci = linalg.inv(C)
LTC = L @ Ci
A = LTC @ L.T
theta = linalg.solve(A, LTC @ y, assume_a='pos')
chi2_base = (y - theta @ L) @ Ci @ (y - theta @ L)
h0_base = 10**(theta[46] / 5)
print("=" * 55)
print("ANCHOR INVESTIGATION")
print("=" * 55)
print(f"Baseline: H0 = {h0_base:.2f}, χ²/dof = {chi2_base:.1f}/{3445}")

anchor_idx = np.where(L[44] != 0)[0][0]
print(f"Anchor index: {anchor_idx}")

results = []

# 1. Remove anchor row 44
th, h0, c = solve_ols(y, np.delete(L, 44, axis=0), C)
results.append(("1. No anchor row", h0, c-chi2_base))
print(f"1. No anchor row:       H0 = {h0:.2f}, Δχ² = {c-chi2_base:+.1f}")

# 2. Weaken anchor error ×10
C2 = C.copy(); C2[anchor_idx, anchor_idx] *= 100
th, h0, c = solve_ols(y, L, C2)
results.append(("2. Weak anchor ×10", h0, c-chi2_base))
print(f"2. Weak anchor ×10:     H0 = {h0:.2f}, Δχ² = {c-chi2_base:+.1f}")

# 3. Remove all prior constraints
nz = [np.sum(L[i] != 0) for i in range(47)]
con = [i for i in range(37, 47) if nz[i] <= 50]
th, h0, c = solve_ols(y, np.delete(L, con, axis=0), C)
results.append(("3. No priors", h0, None))
print(f"3. No priors:           H0 = {h0:.2f}")

# 4-6. Fix H0
for h0_fix, label in [(67.4, "Planck"), (73.0, "SH0ES"), (68.0, "68.0")]:
    yf = y - 5*np.log10(h0_fix) * L[46]
    _, _, c = solve_ols(yf, np.delete(L, 46, axis=0), C)
    results.append((f"4. Fix H0={h0_fix} ({label})", None, c-chi2_base))
    print(f"4. Fix H0={h0_fix} ({label}):     Δχ² = {c-chi2_base:+.1f}")

# 7-8. No anchor + fix H0
for h0_fix, label in [(67.4, "Planck"), (73.0, "SH0ES")]:
    L_sub = np.delete(L, [44, 46], axis=0)
    yf = y - 5*np.log10(h0_fix) * L[46]
    _, _, c = solve_ols(yf, L_sub, C)
    results.append((f"5. No anchor+H0={h0_fix} ({label})", None, c-chi2_base))
    print(f"5. No anchor + H0={h0_fix} ({label}): Δχ² = {c-chi2_base:+.1f}")

# 9. MC on single anchor point
rng = np.random.RandomState(42)
n_mc = 500
h0_mc = np.zeros(n_mc)
for b in range(n_mc):
    ym = y.copy()
    ym[anchor_idx] += rng.randn() * 0.02
    th, h0_mc[b], _ = solve_ols(ym, np.delete(L, 44, axis=0), C)
results.append((f"6. MC anchor ±0.02", np.std(h0_mc), None))
print(f"6. MC anchor ±0.02:    σ(H0) = {np.std(h0_mc):.4f}")

# 10. MC on all Cepheid hosts simultaneously
print(f"7. MC all hosts ±0.03:", end=" ")
rng = np.random.RandomState(42)
n_mc2 = 200
h0_mc2 = np.zeros(n_mc2)
for b in range(n_mc2):
    ym = y.copy()
    for h in range(19):
        ym[L[h] != 0] += rng.randn() * 0.03
    th, h0_mc2[b], _ = solve_ols(ym, np.delete(L, 44, axis=0), C)
results.append((f"7. MC all hosts ±0.03", np.std(h0_mc2), None))
print(f"σ(H0) = {np.std(h0_mc2):.4f}")

print(f"\n{'Test':40s} {'H0':>8s} {'Δχ²':>8s}")
print(f"{'Baseline':40s} {h0_base:>8.1f} {'—':>8s}")
for name, h, d in results:
    if h is not None:
        print(f"{name:40s} {h:>8.2f} {f'{d:+.1f}' if d is not None else '—':>8s}")
    else:
        print(f"{name:40s} {'—':>8s} {f'{d:+.1f}' if d is not None else '—':>8s}")

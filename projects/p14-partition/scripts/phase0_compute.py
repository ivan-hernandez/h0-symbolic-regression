"""Phase 0: Compute partition function p(n) via Euler's recurrence."""
import csv, os, math, sys

NMAX = 100000
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTDIR, exist_ok=True)

def compute_partitions(nmax):
    p = [0] * (nmax + 1)
    p[0] = 1
    for n in range(1, nmax + 1):
        total = 0
        k = 1
        while True:
            g1 = k * (3 * k - 1) // 2  # pentagonal numbers
            if g1 > n:
                break
            sign = 1 if k % 2 == 1 else -1
            total += sign * p[n - g1]
            g2 = k * (3 * k + 1) // 2
            if g2 <= n:
                total += sign * p[n - g2]
            k += 1
        p[n] = total
    return p

print('Computing p(n) for n up to %d...' % NMAX)
p = compute_partitions(NMAX)
print('Done. Largest value: log10(p(%d)) = %.2f' % (NMAX, math.log10(p[NMAX])))

# Save
with open(os.path.join(OUTDIR, 'p_n.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n', 'p_n', 'log10_p_n', 'sqrt_n'])
    for n in range(1, NMAX + 1):
        w.writerow([n, str(p[n]), math.log10(p[n]), math.sqrt(n)])

print('Saved %d values to %s' % (NMAX, os.path.join(OUTDIR, 'p_n.csv')))

# Quick verification: known values
known = {1: 1, 2: 2, 3: 3, 4: 5, 5: 7, 6: 11, 10: 42, 20: 627, 50: 204226, 100: 190569292}
for n, true_val in known.items():
    assert p[n] == true_val, f'Mismatch at n={n}: got {p[n]}, expected {true_val}'
print('Known values verified.')

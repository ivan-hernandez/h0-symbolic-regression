"""Phase 0: Download Pantheon+ SNIa data with all columns."""
import csv, os, math, urllib.request
import numpy as np

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTDIR, exist_ok=True)

URL = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
       "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
       "Pantheon%2BSH0ES.dat")

print('Downloading Pantheon+ data...')
req = urllib.request.urlopen(URL, timeout=60)
lines = req.read().decode().strip().split('\n')
hdr = lines[0].split()
print('Columns:', len(hdr))

# Parse
rows = []
for line in lines[1:]:
    c = line.split()
    if len(c) < len(hdr):
        continue
    row = {}
    for i, col in enumerate(hdr):
        val = c[i]
        try:
            row[col] = float(val)
        except ValueError:
            row[col] = val
    rows.append(row)

print('%d SNe loaded' % len(rows))

# Save as CSV
csv_path = os.path.join(OUTDIR, 'pantheon_plus.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    w.writerows(rows)
print('Saved to %s' % csv_path)

# Quick stats
mu = np.array([r['MU_SH0ES'] for r in rows])
z = np.array([r['zHD'] for r in rows])
x1 = np.array([r['x1'] for r in rows])
c = np.array([r['c'] for r in rows])
host_mass = np.array([r['HOST_LOGMASS'] for r in rows])

print('\n=== Quick stats ===')
print('z range: [%.4f, %.4f]' % (np.min(z), np.max(z)))
print('mu range: [%.2f, %.2f]' % (np.min(mu), np.max(mu)))
print('x1 range: [%.2f, %.2f]' % (np.min(x1), np.max(x1)))
print('c range: [%.3f, %.3f]' % (np.min(c), np.max(c)))
print('Host mass range: [%.2f, %.2f]' % (np.nanmin(host_mass), np.nanmax(host_mass)))
print('SNe with host mass: %d' % np.sum(~np.isnan(host_mass)))

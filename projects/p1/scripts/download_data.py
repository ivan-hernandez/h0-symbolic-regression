"""
Download exoplanet data from NASA Exoplanet Archive (ps table).
"""
import numpy as np
import urllib.request, json

query = """
select pl_name,hostname,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,
pl_rade,pl_radeerr1,pl_radeerr2,discoverymethod,pl_orbper,
pl_eqt,st_teff,st_lum,st_rad,st_mass
from ps
where pl_bmasse is not null and pl_rade is not null and pl_bmasse>0
"""
url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={urllib.request.quote(query)}&format=json"

print("Downloading from NASA Exoplanet Archive...")
req = urllib.request.Request(url)
req.add_header('Accept', 'application/json')
resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode())
print(f"Got {len(data)} rows")

records = []
for row in data:
    try:
        mass = float(row.get('pl_bmasse') or 0)
        rad = float(row.get('pl_rade') or 0)
    except (ValueError, TypeError):
        continue
    if mass <= 0 or rad <= 0:
        continue
    records.append({
        'name': str(row.get('pl_name', '')),
        'host': str(row.get('hostname', '')),
        'mass': mass,
        'mass_err_low': float(row.get('pl_bmasseerr1') or np.nan),
        'mass_err_high': float(row.get('pl_bmasseerr2') or np.nan),
        'rad': rad,
        'rad_err_low': float(row.get('pl_radeerr1') or np.nan),
        'rad_err_high': float(row.get('pl_radeerr2') or np.nan),
        'method': str(row.get('discoverymethod', '')),
        'period': float(row.get('pl_orbper') or np.nan),
        'teq': float(row.get('pl_eqt') or np.nan),
        'st_teff': float(row.get('st_teff') or np.nan),
        'st_lum': float(row.get('st_lum') or np.nan),
        'st_rad': float(row.get('st_rad') or np.nan),
        'st_mass': float(row.get('st_mass') or np.nan),
    })

print(f"Parsed {len(records)} planets with mass+radius.")

dtype = [('name', 'U50'), ('host', 'U50'),
         ('mass', 'f8'), ('mass_err_low', 'f8'), ('mass_err_high', 'f8'),
         ('rad', 'f8'), ('rad_err_low', 'f8'), ('rad_err_high', 'f8'),
         ('method', 'U30'), ('period', 'f8'), ('teq', 'f8'),
         ('st_teff', 'f8'), ('st_lum', 'f8'), ('st_rad', 'f8'), ('st_mass', 'f8')]
arr = np.zeros(len(records), dtype=dtype)
for i, p in enumerate(records):
    arr[i] = (p['name'], p['host'], p['mass'], p['mass_err_low'], p['mass_err_high'],
              p['rad'], p['rad_err_low'], p['rad_err_high'], p['method'],
              p['period'], p['teq'], p['st_teff'], p['st_lum'], p['st_rad'], p['st_mass'])

np.savez_compressed('../data/exoplanets.npz', planets=arr)
print(f"Saved {len(arr)} planets to data/exoplanets.npz")

masses = arr['mass']
rads = arr['rad']
print(f"\nMass: {masses.min():.4f} - {masses.max():.1f} M_Earth  (median {np.median(masses):.2f})")
print(f"Radius: {rads.min():.4f} - {rads.max():.1f} R_Earth  (median {np.median(rads):.2f})")

methods, counts = np.unique(arr['method'], return_counts=True)
print("\nDetection methods:")
for m, c in sorted(zip(methods, counts), key=lambda x: -x[1]):
    print(f"  {m}: {c}")

"""
Download GWTC data from GWOSC event API.
Gets all confident detections with component masses.
"""
import numpy as np
import urllib.request, json, os

url = 'https://www.gw-openscience.org/eventapi/json/GWTC/'
print("Downloading GWTC catalog...")
data = json.loads(urllib.request.urlopen(urllib.request.Request(url)).read().decode())
events_dict = data['events']
print(f"Total events: {len(events_dict)}")

# Parse
events = []
kept = 0
for name, ev in events_dict.items():
    try:
        far = float(ev.get('far', 1))
        m1 = float(ev.get('mass_1_source', 0))
        m2 = float(ev.get('mass_2_source', 0))
    except (ValueError, TypeError):
        continue
    if m1 <= 0 or m2 <= 0:
        continue
    cat = ev.get('catalog.shortName', '')
    p_astro = float(ev.get('p_astro', 1)) if ev.get('p_astro') else 1
    
    events.append({
        'name': ev.get('commonName', name),
        'catalog': cat,
        'm1': m1,
        'm1_low': float(ev.get('mass_1_source_lower', 0)),
        'm1_high': float(ev.get('mass_1_source_upper', 0)),
        'm2': m2,
        'm2_low': float(ev.get('mass_2_source_lower', 0)),
        'm2_high': float(ev.get('mass_2_source_upper', 0)),
        'far': far,
        'snr': float(ev.get('network_matched_filter_snr', 0)),
        'p_astro': p_astro,
        'mchirp': float(ev.get('chirp_mass_source', 0)),
    })

events.sort(key=lambda e: e['m1'], reverse=True)
print(f"Total parsed: {len(events)}")

# Filter to confident GWTC-3 events
confident = [e for e in events if e['far'] < 0.01 and e['p_astro'] > 0.5]
print(f"Confident (FAR<0.01, p_astro>0.5): {len(confident)}")

# Save
os.makedirs('../data', exist_ok=True)
dtype = [('name', 'U30'), ('catalog', 'U20'),
         ('m1', 'f8'), ('m1_low', 'f8'), ('m1_high', 'f8'),
         ('m2', 'f8'), ('m2_low', 'f8'), ('m2_high', 'f8'),
         ('far', 'f8'), ('snr', 'f8'), ('p_astro', 'f8'), ('mchirp', 'f8')]
arr = np.zeros(len(confident), dtype=dtype)
for i, e in enumerate(confident):
    arr[i] = (e['name'], e['catalog'], e['m1'], e['m1_low'], e['m1_high'],
              e['m2'], e['m2_low'], e['m2_high'],
              e['far'], e['snr'], e['p_astro'], e['mchirp'])

np.savez_compressed('../data/gwtc3.npz', events=arr, all_events=confident)
print(f"\nSaved {len(arr)} confident events to data/gwtc3.npz")
print(f"m1: {arr['m1'].min():.1f} - {arr['m1'].max():.1f} M_sun (median {np.median(arr['m1']):.1f})")
print(f"m2: {arr['m2'].min():.1f} - {arr['m2'].max():.1f} M_sun (median {np.median(arr['m2']):.1f})")

# Separate BBH, BNS, NSBH
ns_cut = 3.0
bbh = (arr['m1'] > ns_cut) & (arr['m2'] > ns_cut)
bns = (arr['m1'] <= ns_cut) & (arr['m2'] <= ns_cut)
nsbh = (arr['m1'] > ns_cut) & (arr['m2'] <= ns_cut)
print(f"BBH: {bbh.sum()}, BNS: {bns.sum()}, NSBH: {nsbh.sum()}")

# Catalog breakdown
for cat in np.unique(arr['catalog']):
    cnt = np.sum(arr['catalog'] == cat)
    print(f"  {cat}: {cnt}")

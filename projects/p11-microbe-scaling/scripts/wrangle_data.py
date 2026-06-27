"""
Wrangle all microbial metabolic scaling data sources into a unified format.
Merges Hoehler+2023 (full dataset, 3821 entries, 246 microbes) with 
DeLong+2010 (355 metabolic rate entries, 172 rmax entries).

Normalizes units: mass in g, metabolic rate in W.
Output: unified CSV for Phase 1 exploration.
"""
import os, csv, numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Load Hoehler+2023 complete dataset
# ============================================================
hoehler_path = os.path.join(DATA_DIR, 'hoehler2023_complete.csv')
hoehler_rows = []
# Taxonomic levels we want to clean up
with open(hoehler_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Only take microbes (Archaea + Bacteria) for P11
        if row['Domain'] in ('Archaea', 'Bacteria'):
            # Parse numeric fields
            mass_str = row.get('Mean_Mass', '').strip()
            mr_str = row.get('Mean_Metabolic_Rate', '').strip()
            mr25_str = row.get('Mean_Metabolic_Rate_25', '').strip()
            
            if mass_str and mr_str:
                try:
                    mass = float(mass_str)
                    mr = float(mr_str)
                    mr25 = float(mr25_str) if mr25_str else np.nan
                    
                    # Classify metabolic state from the 'Rate' column
                    rate_lower = row.get('Rate', '').lower()
                    if any(x in rate_lower for x in ['endogenous', 'starved', 'inactive', 'maintenance', 'basal']):
                        state = 'endogenous'
                    elif any(x in rate_lower for x in ['active', 'growing', 'maximum', 'max']):
                        state = 'active'
                    else:
                        state = 'unknown'
                    
                    hoehler_rows.append({
                        'source': 'Hoehler+2023',
                        'domain': 'Bacteria' if row['Domain'] == 'Bacteria' else 'Archaea',
                        'phylum': row.get('Phylum', ''),
                        'class_': row.get('Class', ''),
                        'order_': row.get('Order', ''),
                        'family': row.get('Family', ''),
                        'genus': row.get('Genus', ''),
                        'species': row.get('Species', ''),
                        'state': state,
                        'mass_g': mass,
                        'metabolic_rate_W': mr,
                        'metabolic_rate_25C_W': mr25,
                        'mass_specific_W_g': float(row.get('Mean_Mass_Specific', 0)) if row.get('Mean_Mass_Specific', '').strip() else np.nan,
                        'mass_specific_25C_W_g': float(row.get('Mean_Mass_Specific_25', 0)) if row.get('Mean_Mass_Specific_25', '').strip() else np.nan,
                    })
                except (ValueError, TypeError):
                    pass

print(f"Hoehler+2023 microbial entries: {len(hoehler_rows)}")

# ============================================================
# 2. Load DeLong+2010 metabolic rate data
# ============================================================
delong_path = os.path.join(DATA_DIR, 'delong2010_metabolic_rates.csv')
delong_rows = []
with open(delong_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Map group to domain
        group = row['group'].strip().lower()
        if group == 'prokaryote':
            domain = 'Bacteria'  # DeLong data doesn't distinguish Archaea
        elif group == 'protist':
            domain = 'Eukaryota'
        else:
            domain = 'Eukaryota'  # metazoans
            
        delong_rows.append({
            'source': 'DeLong+2010',
            'domain': domain,
            'phylum': '',
            'class_': '',
            'order_': '',
            'family': '',
            'genus': row['species'].split()[0] if row['species'] else '',
            'species': row['species'],
            'state': row['state'],
            'mass_g': float(row['mass_g']),
            'metabolic_rate_W': float(row['metabolic_rate_W']),
            'metabolic_rate_25C_W': np.nan,  # not available
            'mass_specific_W_g': np.nan,
            'mass_specific_25C_W_g': np.nan,
        })

print(f"DeLong+2010 metabolic rate entries: {len(delong_rows)}")
prok = [r for r in delong_rows if r['domain'] == 'Bacteria']
prot = [r for r in delong_rows if r['domain'] == 'Eukaryota']
meta = []  # no separate tracking needed
print(f"  Prokaryotes: {len(prok)}, Eukaryotes (incl protists): {len(prot)}")

# ============================================================
# 3. Merge
# ============================================================
merged = hoehler_rows + delong_rows

# Add metadata
for row in merged:
    # Data type for analysis
    if row['domain'] in ('Archaea', 'Bacteria'):
        row['group'] = 'prokaryote'
    elif row['domain'] == 'Eukaryota':
        # Check if unicellular (protist) or multicellular
        if row['source'] == 'DeLong+2010':
            # From Delong, protists are explicitly labelled by group
            pass  # Already have group in source
        row['group'] = 'eukaryote_unknown'  # will be refined

# Standardize output fields
fieldnames = ['source', 'domain', 'group', 'phylum', 'class_', 'order_', 'family', 'genus', 
              'species', 'state', 'mass_g', 'metabolic_rate_W', 'metabolic_rate_25C_W',
              'mass_specific_W_g', 'mass_specific_25C_W_g']

# Filter for analysis
out_path = os.path.join(OUT_DIR, 'microbial_metabolic_data.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(merged)

print(f"\nMerged dataset: {len(merged)} total entries")
print(f"  Hoehler+2023 microbes: {len(hoehler_rows)}")
print(f"  DeLong+2010: {len(delong_rows)}")
print(f"  Domains: {set(r['domain'] for r in merged)}")
print(f"  States: {set(r['state'] for r in merged)}")
print(f"  Mass range: {min(r['mass_g'] for r in merged):.2e} to {max(r['mass_g'] for r in merged):.2e} g")
print(f"  MR range: {min(r['metabolic_rate_W'] for r in merged):.2e} to {max(r['metabolic_rate_W'] for r in merged):.2e} W")
print(f"\nSaved to {out_path}")

# ============================================================
# 4. Quick OLS exploration
# ============================================================
from numpy.polynomial import Polynomial

print("\n--- Quick OLS exponents by group ---")
for group_name in ['prokaryote']:
    subset = [r for r in merged if r['group'] == group_name or r['domain'] in ('Archaea', 'Bacteria')]
    if not subset:
        continue
    masses = np.array([r['mass_g'] for r in subset])
    mrs = np.array([r['metabolic_rate_W'] for r in subset])
    logM = np.log10(masses)
    logB = np.log10(mrs)
    
    # OLS fit
    A = np.vstack([np.ones_like(logM), logM]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, logB, rcond=None)
    logB_pred = A @ coeffs
    resid = logB - logB_pred
    mse = np.mean(resid**2)
    
    print(f"  {group_name} ({len(subset)} pts): B = {10**coeffs[0]:.2e} * M^{coeffs[1]:.3f}, RMSE={np.sqrt(mse):.3f} dex")

# Also do by source
for source_name in ['Hoehler+2023', 'DeLong+2010']:
    subset = [r for r in merged if r['source'] == source_name]
    masses = np.array([r['mass_g'] for r in subset])
    mrs = np.array([r['metabolic_rate_W'] for r in subset])
    logM = np.log10(masses)
    logB = np.log10(mrs)
    A = np.vstack([np.ones_like(logM), logM]).T
    coeffs, *_ = np.linalg.lstsq(A, logB, rcond=None)
    print(f"  {source_name}: B = {10**coeffs[0]:.2e} * M^{coeffs[1]:.3f}")

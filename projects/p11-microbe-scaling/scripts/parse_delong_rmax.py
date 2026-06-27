"""
Parse DeLong+2010 rmax data from Section III supplementary materials.
"""
import re, os, csv

PDF_TEXT = '/tmp/delong_full.txt'
OUT_DIR = '/home/ivan/general-conversation/projects/p11-microbe-scaling/data/'

with open(PDF_TEXT) as f:
    lines = [l.rstrip('\n') for l in f.readlines()]

def parse_rmax_table(lines, start_idx, end_idx, group):
    """Parse rmax table: Species, Body mass (g), rmax (day-1)"""
    rows = []
    i = start_idx
    while i < end_idx:
        line = lines[i].strip()
        if not line or line.startswith('Source for') or line == 'IIIc. Metazoans':
            i += 1
            continue
        
        # Split by 2+ spaces
        parts = re.split(r'  +', line)
        
        # Filter out headers and page numbers
        if 'Species' in line or 'Body mass' in line or line.isdigit():
            i += 1
            continue
        
        if len(parts) >= 3:
            species = parts[0].strip()
            mass = parts[1].strip()
            rmax = parts[2].strip()
            
            # Validate
            if re.match(r'^-?\d+\.?\d*e[+-]?\d*$', mass, re.I) and \
               re.match(r'^-?\d+\.?\d*e?[+-]?\d*$', rmax, re.I):
                rows.append({
                    'group': group,
                    'species': species,
                    'mass_g': float(mass),
                    'rmax_day': float(rmax)
                })
            elif species and mass:
                # Maybe species name has spaces and spans multiple columns
                pass
        elif len(parts) >= 2 and len(parts) < 3:
            # Might be a continuation line - skip
            pass
        
        i += 1
    
    csv_path = os.path.join(OUT_DIR, f'delong_rmax_{group}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'species', 'mass_g', 'rmax_day'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {csv_path}")
    return rows

# Find section III boundaries
raw_lines = lines
iii_start = next(i for i, l in enumerate(raw_lines) if 'IVa. Prokaryotes' in l)
iii_end_prok = next(i for i, l in enumerate(raw_lines) if 'IVb. Protists' in l)
iii_start_prot = iii_end_prok
iii_end_prot = next(i for i, l in enumerate(raw_lines) if 'IVc. Metazoans' in l)
iii_start_meta = iii_end_prot
iii_end = len(raw_lines)

print(f"IVa Prokaryotes: lines {iii_start}-{iii_end_prok}")
print(f"IVb Protists: lines {iii_start_prot}-{iii_end_prot}")
print(f"IVc Metazoans: lines {iii_start_meta}-end")

prok = parse_rmax_table(lines, iii_start + 5, iii_end_prok, 'prokaryote')
prot = parse_rmax_table(lines, iii_start_prot + 5, iii_end_prot, 'protist')
meta = parse_rmax_table(lines, iii_start_meta + 5, iii_end, 'metazoan')

all_rows = prok + prot + meta
all_path = os.path.join(OUT_DIR, 'delong2010_rmax.csv')
with open(all_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['group', 'species', 'mass_g', 'rmax_day'])
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nTotal rmax: {len(all_rows)} rows")
for g in ['prokaryote', 'protist', 'metazoan']:
    cnt = len([r for r in all_rows if r['group'] == g])
    print(f"  {g}: {cnt}")

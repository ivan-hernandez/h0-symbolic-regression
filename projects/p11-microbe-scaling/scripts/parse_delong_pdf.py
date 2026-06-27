"""
Parse DeLong+2010 supplementary data from PDF text extraction.
Extracts: IIa (Prokaryotes), IIb (Protists), IIc (Metazoans) metabolic rates,
and III (rmax) data.
"""
import re, os, csv

PDF_TEXT = '/tmp/delong_full.txt'
OUT_DIR = '/home/ivan/general-conversation/projects/p11-microbe-scaling/data/'

with open(PDF_TEXT) as f:
    lines = [l.rstrip('\n') for l in f.readlines()]

# Remove form-feed lines
lines = [l for l in lines if l != '\x0c']

def parse_two_column_table(lines, start_idx, end_idx, filename, group):
    """Parse two-column table: left=endogenous, right=active. Each row has columns on consecutive lines."""
    species_col = 0
    rows = []
    i = start_idx
    while i < end_idx:
        line = lines[i].strip()
        # Skip empty/short lines and page numbers
        if not line or line.isdigit() or line.startswith('Source for') or line.startswith('Sources for'):
            i += 1
            continue

        # Try to parse as a data row
        # Pattern: species name (may span multiple lines) + mass + rate + species + mass + rate
        # Split on double or more spaces to get columns
        parts = re.split(r'  +', line)
        
        if len(parts) >= 6:
            # Check if this is a metadata line or data line
            col0 = parts[0].strip()
            col3 = parts[3].strip() if len(parts) > 3 else ''
            
            # Skip header lines containing "Species" or "Metabolic"
            if col0 and not col0[0].isupper():
                pass  # Might be a species name starting with lowercase
            
            # Extract the data
            left_species = parts[0].strip()
            left_mass = parts[1].strip()
            left_rate = parts[2].strip()
            right_species = parts[3].strip()
            right_mass = parts[4].strip() if len(parts) > 4 else ''
            right_rate = parts[5].strip() if len(parts) > 5 else ''
            
            # Validate - check if mass looks like scientific notation
            def is_mass(val):
                return bool(re.match(r'^-?\d+\.?\d*e[+-]\d+$', val, re.I))
            
            left_mass_ok = is_mass(left_mass)
            right_mass_ok = right_mass and is_mass(right_mass)
            
            if left_mass_ok or right_mass_ok:
                if left_mass_ok:
                    rows.append({
                        'group': group, 'state': 'endogenous',
                        'species': left_species, 'mass_g': float(left_mass),
                        'metabolic_rate_W': float(left_rate)
                    })
                if right_mass_ok and right_species:
                    rows.append({
                        'group': group, 'state': 'active',
                        'species': right_species, 'mass_g': float(right_mass),
                        'metabolic_rate_W': float(right_rate)
                    })
        elif len(parts) == 3 and i > start_idx + 5:
            # Single column (endogenous only) - check if previous line had right side
            prev_line = lines[i-1].strip() if i > 0 else ''
            prev_parts = re.split(r'  +', prev_line)
            
            col0 = parts[0].strip()
            col1 = parts[1].strip()
            col2 = parts[2].strip()
            
            def is_mass(val):
                return bool(re.match(r'^-?\d+\.?\d*e[+-]\d+$', val, re.I))
            
            if is_mass(col1) and not is_mass(col2):
                # This row is the continuation of a left-side species name
                pass
            elif is_mass(col1) and is_mass(col2):
                # Left side data
                species = col0
                if species[0].isupper() or species[0].islower():
                    rows.append({
                        'group': group, 'state': 'endogenous',
                        'species': species, 'mass_g': float(col1),
                        'metabolic_rate_W': float(col2)
                    })
        elif len(parts) >= 4:
            # Might be a row where left side fills all 4 parts  
            col0 = parts[0].strip()
            col1 = parts[1].strip()
            col2 = parts[2].strip()
            col3 = parts[3].strip()
            
            def is_mass(val):
                return bool(re.match(r'^-?\d+\.?\d*e[+-]\d+$', val, re.I))
            
            # Try: left_species left_mass left_rate right_species
            # or: left_species_cont left_mass left_rate right_cont
            if is_mass(col1) and is_mass(col2):
                # Left side complete
                # Check if col3 is a species name
                if not is_mass(col3) and col3 and (col3[0].isupper() or col3[0].islower()):
                    rows.append({
                        'group': group, 'state': 'endogenous',
                        'species': col0, 'mass_g': float(col1),
                        'metabolic_rate_W': float(col2)
                    })
                    # right species continues on next line
                elif is_mass(col3):
                    rows.append({
                        'group': group, 'state': 'endogenous',
                        'species': col0, 'mass_g': float(col1),
                        'metabolic_rate_W': float(col2)
                    })
                    # right: species unknown, mass=col3, rate on next line
        elif len(parts) == 2:
            col0 = parts[0].strip()
            col1 = parts[1].strip()
            if col0 and col1 and col0[0].isupper():
                # Might be a continuation line - skip
                pass

        i += 1
    
    # Save
    csv_path = os.path.join(OUT_DIR, filename)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'state', 'species', 'mass_g', 'metabolic_rate_W'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {csv_path}")
    return rows


# Find section boundaries using line numbers (1-indexed in the file)
def find_section(text_lines, start_marker, end_marker=None):
    """Find start/end line indices for a section"""
    start = None
    for i, line in enumerate(text_lines):
        if start_marker in line:
            start = i
            break
    if start is None:
        return None, None
    
    if end_marker:
        for i in range(start + 1, len(text_lines)):
            if end_marker in text_lines[i]:
                return start, i
    
    return start, len(text_lines)


# Find all sections
# Read the raw text without line filtering for section markers
with open(PDF_TEXT) as f:
    raw_lines = f.readlines()

# IIa. Prokaryotes (Metabolic rate data)
ii_start = next(i for i, l in enumerate(raw_lines) if 'IIa. Prokaryotes' in l)
ii_end = next(i for i, l in enumerate(raw_lines) if 'IIb. Protists' in l)
print(f"IIa Prokaryotes: lines {ii_start}-{ii_end}")

# Parse prokaryote data
prok_rows = parse_two_column_table(lines, ii_start + 5, ii_end, 'delong_prokaryotes.csv', 'prokaryote')

# IIb. Protists
iiib_start = next(i for i, l in enumerate(raw_lines) if 'IIb. Protists' in l)
iiib_end = next(i for i, l in enumerate(raw_lines) if 'IIc. Metazoans' in l)
print(f"IIb Protists: lines {iiib_start}-{iiib_end}")

prot_rows = parse_two_column_table(lines, iiib_start + 5, iiib_end, 'delong_protists.csv', 'protist')

# IIc. Metazoans 
iic_start = next(i for i, l in enumerate(raw_lines) if 'IIc. Metazoans' in l)
iic_end = next(i for i, l in enumerate(raw_lines) if 'Supplementary Online Materials III' in l)
print(f"IIc Metazoans: lines {iic_start}-{iic_end}")

meta_rows = parse_two_column_table(lines, iic_start + 5, iic_end, 'delong_metazoans.csv', 'metazoan')

# Combine all
all_rows = prok_rows + prot_rows + meta_rows
all_path = os.path.join(OUT_DIR, 'delong2010_metabolic_rates.csv')
with open(all_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['group', 'state', 'species', 'mass_g', 'metabolic_rate_W'])
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nTotal: {len(all_rows)} rows across all groups")
for g in ['prokaryote', 'protist', 'metazoan']:
    grp = [r for r in all_rows if r['group'] == g]
    for s in ['endogenous', 'active']:
        subset = [r for r in grp if r['state'] == s]
        print(f"  {g} {s}: {len(subset)}")

#!/usr/bin/env python3
"""Explore the compact object mass distribution around 1-10 Msun."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.load_masses import load_all
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = load_all()
m = d['mass']

# Focus on 1-10 Msun (NS-BH transition)
mask = (m >= 0.8) & (m <= 10)
m_lo = m[mask]

print(f"Total objects: {len(m)}")
print(f"In 0.8-10 Msun range: {len(m_lo)}")

print("\nMass function features:")
for s in set(d['source']):
    sub = d[(d['source'] == s) & mask]
    print(f"  {s}: {len(sub)} objects, masses {sub['mass'].min():.2f}-{sub['mass'].max():.2f}")

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
bins = np.linspace(0.8, 5, 42)
ax.hist(m_lo, bins=bins, alpha=0.7, color='#1a73e8', edgecolor='white')
ax.axvline(2.3, color='red', ls='--', lw=1.5, label='TOV limit (~2.3)')
ax.axvline(3.0, color='orange', ls=':', lw=1.5, label='min BH (GW)')
ax.axvline(5.0, color='purple', ls=':', lw=1.5, label='min BH (XRB)')
ax.set(xlabel='Mass (Msun)', ylabel='Count', title='0.8-5 Msun', yscale='log')
ax.legend(fontsize=7)

ax = axes[1]
bins = np.linspace(0.8, 10, 60)
ax.hist(m_lo, bins=bins, alpha=0.7, color='#1a73e8', edgecolor='white')
ax.axvline(2.3, color='red', ls='--', lw=1.5, label='TOV limit')
ax.set(xlabel='Mass (Msun)', ylabel='Count', title='0.8-10 Msun', yscale='log')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('/home/ivan/general-conversation/projects/p3/analysis/mass_dist.png', dpi=150)
print("\nSaved analysis/mass_dist.png")

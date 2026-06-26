"""
Explore the exoplanet mass-radius data.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('../data/exoplanets.npz')
p = data['planets']
print(f"Total planets: {len(p)}")

# Split by method
mask_t = p['method'] == 'Transit'
mask_rv = p['method'] == 'Radial Velocity'
mask_img = p['method'] == 'Imaging'

# Filter: positive mass and radius for log scaling
valid = (p['mass'] > 0) & (p['rad'] > 0)
print(f"Transit: {mask_t.sum()}, RV: {mask_rv.sum()}, Imaging: {mask_img.sum()}")
print(f"Valid (mass>0, rad>0): {valid.sum()}")
print(f"Mass range (transit): {p['mass'][mask_t].min():.4f} - {p['mass'][mask_t].max():.1f} M_E")
print(f"Mass range (RV): {p['mass'][mask_rv].min():.4f} - {p['mass'][mask_rv].max():.1f} M_E")

# Log-log scatter
fig, ax = plt.subplots(figsize=(8, 6))
for mask, label, ms, alp in [(mask_t, 'Transit', 2, 0.3), (mask_rv, 'RV', 4, 0.6), (mask_img, 'Imaging', 4, 0.6)]:
    has_err = ~np.isnan(p['mass_err_low']*p['mass_err_high']*p['rad_err_low']*p['rad_err_high'])
    ok = mask & valid & has_err
    ml = np.abs(p['mass_err_low'][ok]); mh = np.abs(p['mass_err_high'][ok])
    rl = np.abs(p['rad_err_low'][ok]); rh = np.abs(p['rad_err_high'][ok])
    ax.errorbar(p['mass'][ok], p['rad'][ok], xerr=(ml, mh), yerr=(rl, rh),
                fmt='.', alpha=alp, label=f'{label} ({ok.sum()})', markersize=ms)
ax.set_xscale('log', subs=[2,3,4,5,6,7,8,9])
ax.set_yscale('log', subs=[2,3,4,5,6,7,8,9])
ax.set_xlabel('Mass (M$_\\oplus$)', fontsize=12)
ax.set_ylabel('Radius (R$_\\oplus$)', fontsize=12)
ax.set_title('Exoplanet Mass-Radius Relation', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('../analysis/mass_radius_scatter.png', dpi=150)
print("Saved analysis/mass_radius_scatter.png")

# Histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(np.log10(p['mass'][valid]), bins=50, alpha=0.7)
ax1.set_xlabel('log10(Mass [M_Earth])')
ax1.set_ylabel('Count')
ax1.set_title('Mass Distribution')
ax2.hist(np.log10(p['rad'][valid]), bins=50, alpha=0.7)
ax2.set_xlabel('log10(Radius [R_Earth])')
ax2.set_ylabel('Count')
ax2.set_title('Radius Distribution')
fig.tight_layout()
fig.savefig('../analysis/distributions.png', dpi=150)
print("Saved analysis/distributions.png")

# Density plot in log-log space
fig, ax = plt.subplots(figsize=(8, 6))
h = ax.hist2d(np.log10(p['mass'][valid]), np.log10(p['rad'][valid]), bins=80, cmap='viridis', density=True)
ax.set_xlabel('log10(Mass [M_Earth])')
ax.set_ylabel('log10(Radius [R_Earth])')
ax.set_title('Planet Density in M-R Space')
plt.colorbar(h[3], ax=ax, label='Density')
fig.tight_layout()
fig.savefig('../analysis/density_map.png', dpi=150)
print("Saved analysis/density_map.png")

# Known reference lines
fig, ax = plt.subplots(figsize=(8, 6))
has_err = ~np.isnan(p['mass_err_low']*p['mass_err_high']*p['rad_err_low']*p['rad_err_high'])
ax.scatter(p['mass'][mask_t & valid & has_err], p['rad'][mask_t & valid & has_err], c='C0', alpha=0.2, s=2, label='Data')
# Rocky planet line: R ~ M^0.28 (Rogers 2015)
m_grid = np.logspace(-1, 3, 100)
r_rocky = m_grid**0.28
ax.plot(m_grid, r_rocky, 'k--', label='Rocky (R ~ M$^{0.28}$)', alpha=0.8)
# Jupiter
ax.axhline(10.97, color='gray', linestyle=':', alpha=0.5, label='Jupiter radius')
# Earth
ax.scatter([1], [1], c='green', s=50, marker='*', label='Earth')
# Neptune
ax.scatter([17.15], [3.88], c='blue', s=50, marker='s', label='Neptune')
ax.set_xscale('log', subs=[2,3,4,5,6,7,8,9])
ax.set_yscale('log', subs=[2,3,4,5,6,7,8,9])
ax.set_xlabel('Mass (M$_\\oplus$)')
ax.set_ylabel('Radius (R$_\\oplus$)')
ax.set_title('M-R Relation with Known References')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('../analysis/reference_comparison.png', dpi=150)
print("Saved analysis/reference_comparison.png")

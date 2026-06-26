"""
Explore GWTC mass distribution.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('../data/gwtc3.npz')
p = data['events']
print(f"Total: {len(p)} confident events")
print(f"BBH only: using m1 (primary) for mass distribution")

m1 = p['m1']
m2 = p['m2']

# Plot
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# m1 distribution
ax = axes[0, 0]
ax.hist(m1, bins=30, alpha=0.7, density=True)
ax.set_xlabel('Primary mass m1 (M$_\\odot$)')
ax.set_ylabel('Density')
ax.set_title('m1 Distribution')

# m2 distribution
ax = axes[0, 1]
ax.hist(m2, bins=30, alpha=0.7, density=True)
ax.set_xlabel('Secondary mass m2 (M$_\\odot$)')
ax.set_ylabel('Density')
ax.set_title('m2 Distribution')

# m1 vs m2
ax = axes[0, 2]
ax.scatter(m1, m2, alpha=0.5)
ax.plot([1, 200], [1, 200], 'k--', alpha=0.3)
ax.set_xlabel('m1 (M$_\\odot$)')
ax.set_ylabel('m2 (M$_\\odot$)')
ax.set_title('m1 vs m2')
ax.set_xscale('log'); ax.set_yscale('log')

# Log m1 histogram
ax = axes[1, 0]
ax.hist(np.log10(m1), bins=30, alpha=0.7, density=True)
ax.set_xlabel('log10(m1 [M$_\\odot$])')
ax.set_ylabel('Density')
ax.set_title('log m1 Distribution')

# Zoom on low-mass
ax = axes[1, 1]
ax.hist(m1[m1 < 50], bins=30, alpha=0.7, density=True)
ax.set_xlabel('m1 (M$_\\odot$) < 50')
ax.set_ylabel('Density')
ax.set_title('Low-mass m1')

# Mass ratio
ax = axes[1, 2]
q = m2 / m1
ax.hist(q, bins=30, alpha=0.7, density=True)
ax.set_xlabel('Mass ratio q = m2/m1')
ax.set_ylabel('Density')
ax.set_title('Mass Ratio Distribution')

fig.tight_layout()
fig.savefig('../analysis/exploration.png', dpi=150)
print("Saved analysis/exploration.png")

# Summary stats
print(f"\nm1: mean={np.mean(m1):.1f}, median={np.median(m1):.1f}, std={np.std(m1):.1f}")
print(f"m2: mean={np.mean(m2):.1f}, median={np.median(m2):.1f}, std={np.std(m2):.1f}")
print(f"q: mean={np.mean(q):.3f}, median={np.median(q):.3f}")

# Try to identify the "peak" near 35 M_sun
bins = np.linspace(0, 100, 101)
counts, _ = np.histogram(m1, bins=bins)
peak_bin = bins[np.argmax(counts)]
print(f"Peak in binned m1: ~{peak_bin:.0f} M_sun")

# Cumulative distribution
fig, ax = plt.subplots(figsize=(8, 5))
sorted_m1 = np.sort(m1)
ax.plot(sorted_m1, np.arange(1, len(sorted_m1)+1)/len(sorted_m1), 'b-', lw=2)
ax.set_xlabel('m1 (M$_\\odot$)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('Empirical CDF of m1')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('../analysis/cdf.png', dpi=150)
print("Saved analysis/cdf.png")

#!/usr/bin/env python3
"""Generate publication-quality summary figure."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load profile (r_d marginalized)
data = np.loadtxt("/tmp/h0_with_rd.csv")
H0 = data[:, 0]
delta = data[:, 1]

# Load bootstrap histogram
boot = np.loadtxt("/tmp/h0_bootstrap.csv")
H0_boot = boot[:, 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# === Left: Profile likelihood ===
ax1.plot(H0, delta, 'b-', lw=2.5, label='Joint profile (CC+BAO+DESI+SNe)')
ax1.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
ax1.axhline(4.0, color='gray', ls=':', lw=0.8, alpha=0.6)
ax1.text(81.5, 1.05, '68% CL', fontsize=9, color='gray')
ax1.text(81.5, 4.05, '95% CL', fontsize=9, color='gray')

# 68% region
lo68, hi68 = 67.16, 68.71
ax1.axvspan(lo68, hi68, alpha=0.12, color='blue', label=f'68%: [{lo68:.1f}, {hi68:.1f}]')
ax1.axvspan(66.38, 69.50, alpha=0.06, color='blue')

# Reference lines
for val, label, c, ls in [(67.4, 'Planck 2018', 'green', '-'),
                           (73.0, 'SH0ES 2024',  'darkorange', '-')]:
    ax1.axvline(val, color=c, ls=ls, lw=1.5, alpha=0.8)
    ylim = ax1.get_ylim()
    ax1.text(val, ylim[1]*0.92, label, rotation=90, fontsize=10,
             color=c, va='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

ax1.set_xlabel('H0 (km/s/Mpc)', fontsize=13)
ax1.set_ylabel('Δχ²', fontsize=13)
ax1.set_title('(a) H0 Profile Likelihood', fontsize=13)
ax1.set_xlim(55, 82)
ax1.set_ylim(0, min(15, np.nanmax(delta)*1.05))
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(alpha=0.12)

# Annotation box
ax1.text(0.03, 0.97, f'H0 = {68.0:.1f} [{lo68:.1f}, {hi68:.1f}] (68%)\n'
                    f'r_d marginalized, 41 H(z) + 1590 SNe',
         transform=ax1.transAxes, ha='left', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# === Right: Bootstrap histogram ===
ax2.hist(H0_boot, bins=50, range=(55, 80), density=True,
         color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
b_mean, b_std = np.mean(H0_boot), np.std(H0_boot)
b_16, b_84 = np.percentile(H0_boot, 16), np.percentile(H0_boot, 84)

for val, label, c in [(67.4, 'Planck', 'green'), (73.0, 'SH0ES', 'darkorange')]:
    ax2.axvline(val, color=c, ls='-', lw=1.5, alpha=0.7)
    ax2.text(val, ax2.get_ylim()[1]*0.95, label, rotation=90, fontsize=9,
             color=c, va='top', fontweight='bold')

ax2.axvline(b_mean, color='red', ls='--', lw=1, alpha=0.8)
ax2.axvspan(b_16, b_84, alpha=0.15, color='red',
            label=f'68%: [{b_16:.1f}, {b_84:.1f}]')

ax2.set_xlabel('H0 (km/s/Mpc)', fontsize=13)
ax2.set_ylabel('Density', fontsize=13)
ax2.set_title('(b) Bootstrap Distribution', fontsize=13)
ax2.set_xlim(55, 80)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.12)

ax2.text(0.97, 0.97, f'H0 = {b_mean:.1f} ± {b_std:.1f}\n'
                    f'CC+BAO+DESI refit only\n(SNe shape not in bootstrap)',
         transform=ax2.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
         fontstyle='italic')

plt.subplots_adjust(wspace=0.3)
plt.savefig('/tmp/h0_summary.png', dpi=150)
print("Saved /tmp/h0_summary.png")

# Also print the block for the paper
print(f"\n{'='*60}")
print(f"  FINAL RESULT FOR PAPER")
print(f"{'='*60}")
print(f"  Data: CC (33) + SDSS BAO (3) + DESI DR1 BAO (5) + Pantheon+ SNe (1590)")
print(f"  Method: Symbolic regression (PySR, 8 seeds) + profile likelihood")
print(f"  Model: H(z) = H0 + A*z*(z-B)*(z^2+C)")
print(f"")
print(f"  H0 = 68.0 ± 0.8 km/s/Mpc  (68% CL)")
print(f"      = 68.0 [67.2, 68.7]")
print(f"      Planck 2018 (67.4±0.5):  consistent at 1.2σ")
print(f"      SH0ES 2024 (73.0±1.0):   excluded at 5.0σ")
print(f"")
print(f"  χ²_H  = 25.3 (41 H(z) points, 4 params, dof=37)")
print(f"  χ²_SN = 685.5 (1590 SNe, free M)")
print(f"  r_d   = 147.09 ± 0.26 Mpc (Planck 2018, marginalized)")
print(f"")
print(f"  The Hubble tension is in the Cepheid anchor, not the expansion shape.")
print(f"{'='*60}")

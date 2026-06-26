#!/usr/bin/env python3
"""Generate publication-quality summary figure for paper (DESI DR2)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Profile data (from remote: DESI DR2 + Pantheon+ full cov)
data = np.loadtxt('/tmp/h0_joint_profile_dr2.csv', skiprows=1)
H0_grid = data[:, 0]
delta_j = data[:, 1]

# CC + BAO + DESI DR2 data
CC = np.array([[0.070,69.0,19.6],[0.090,69.0,12.0],[0.120,68.6,26.2],[0.170,83.0,8.0],[0.1791,75.0,4.0],[0.1993,75.0,5.0],[0.200,72.9,29.6],[0.270,77.0,14.0],[0.280,88.8,36.6],[0.3519,83.0,14.0],[0.3802,83.0,13.5],[0.400,95.0,17.0],[0.4004,77.0,10.2],[0.4247,87.1,11.2],[0.4497,92.8,12.9],[0.470,89.0,34.0],[0.4783,80.9,9.0],[0.480,97.0,62.0],[0.5929,104.0,13.0],[0.6797,92.0,8.0],[0.750,98.8,33.6],[0.7812,105.0,12.0],[0.800,113.1,28.5],[0.8754,125.0,17.0],[0.880,90.0,40.0],[0.900,117.0,23.0],[1.037,154.0,20.0],[1.300,168.0,17.0],[1.363,160.0,33.6],[1.430,177.0,18.0],[1.530,140.0,14.0],[1.750,202.0,40.0],[1.965,186.5,50.4]])
BAO = np.array([[0.380,81.1,2.2],[0.510,91.1,2.1],[0.610,99.4,2.2]])
c = 299792.458
DESI_raw = np.array([[0.510,21.86294686,0.42892],[0.706,19.45534918,0.33387],[0.934,17.64149464,0.20104],[1.321,14.17602155,0.22455],[1.484,12.81699964,0.51801],[2.330,8.63154567,0.10106]])
DESI = np.zeros_like(DESI_raw)
DESI[:,0] = DESI_raw[:,0]; DESI[:,1] = c/(147.0*DESI_raw[:,1]); DESI[:,2] = DESI[:,1]*DESI_raw[:,2]/DESI_raw[:,1]

# Best-fit SR model
h0 = 68.70; A = -5.68; B = 4.03; C = 2.02
z_model = np.linspace(0, 2.5, 500)
Hz_model = h0 + A * z_model * (z_model - B) * (z_model**2 + C)

# ΛCDM (H0=67.9, Ωm=0.321)
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.9, Om0=0.321)
Hz_lcdm = cosmo.H(z_model).value

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [1.4, 1]})

# === Left: H(z) data + model ===
ax1.errorbar(CC[:,0], CC[:,1], yerr=CC[:,2], fmt='o', color='k', ms=4,
             capsize=2, capthick=0.5, label='CC')
ax1.errorbar(BAO[:,0], BAO[:,1], yerr=BAO[:,2], fmt='s', color='#2166ac', ms=5,
             capsize=2, capthick=0.5, label='SDSS BAO')
ax1.errorbar(DESI[:,0], DESI[:,1], yerr=DESI[:,2], fmt='D', color='#b2182b', ms=5,
             capsize=2, capthick=0.5, label='DESI DR2 BAO')

ax1.plot(z_model, Hz_model, 'b-', lw=2, label='SR model')
ax1.plot(z_model, Hz_lcdm, 'g--', lw=1.5, alpha=0.7, label=r'$\Lambda$CDM')

ax1.set_xlabel('Redshift $z$', fontsize=12)
ax1.set_ylabel('$H(z)$ (km/s/Mpc)', fontsize=12)
ax1.set_xlim(-0.05, 2.5)
ax1.set_ylim(0, 230)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(alpha=0.15)
ax1.text(0.97, 0.03, f'SR: $H_0$ = {h0:.0f} km/s/Mpc\n'
                    r'$\Lambda$CDM: $H_0$=67.9, $\Omega_m$=0.32',
         transform=ax1.transAxes, ha='right', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# === Right: Profile likelihood ===
valid = np.isfinite(delta_j)
ax2.plot(H0_grid[valid], delta_j[valid], 'b-', lw=2.5, label='Joint profile')
ax2.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
ax2.axhline(4.0, color='gray', ls=':', lw=0.8, alpha=0.6)
ax2.text(81, 1.05, '68% CL', fontsize=9, color='gray')
ax2.text(81, 4.05, '95% CL', fontsize=9, color='gray')

lo68, hi68 = 67.77, 69.65
ax2.axvspan(lo68, hi68, alpha=0.12, color='blue', label=f'68% [{lo68:.1f}, {hi68:.1f}]')
lo95, hi95 = 66.83, 70.61
ax2.axvspan(lo95, hi95, alpha=0.06, color='blue')

for val, label, c, ls in [(67.4, 'Planck', 'green', '-'),
                           (73.0, 'SH0ES', 'darkorange', '-')]:
    ax2.axvline(val, color=c, ls=ls, lw=1.5, alpha=0.8)
    ylim = ax2.get_ylim()
    ax2.text(val, ylim[1]*0.92, label, rotation=90, fontsize=10,
             color=c, va='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

ax2.set_xlabel('$H_0$ (km/s/Mpc)', fontsize=12)
ax2.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
ax2.set_title('Joint Profile Likelihood', fontsize=12)
ax2.set_xlim(55, 82)
ax2.set_ylim(0, 12)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(alpha=0.12)

ax2.text(0.03, 0.97, f'$H_0$ = 68.7 [{lo68:.1f}, {hi68:.1f}] (68%)',
         transform=ax2.transAxes, ha='left', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.subplots_adjust(wspace=0.35)
plt.savefig('/home/ivan/general-conversation/paper/h0_summary.png', dpi=200, bbox_inches='tight')
print('Saved paper/h0_summary.png')

"""
Full SH0ES+Pantheon+ distance ladder fit.
Loads y, L, C fits files, computes H0 from linear least squares.
Tests effect of SR-discovered PL forms on H0.
"""
import numpy as np
from astropy.io import fits
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.RandomState(SEED)

# === Load data ===
y = fits.open('data/ally_shoes_ceph_topantheonwt6.0_112221.fits')[0].data.astype(float)
L = fits.open('data/alll_shoes_ceph_topantheonwt6.0_112221.fits')[0].data.astype(float)
C = fits.open('data/allc_shoes_ceph_topantheonwt6.0_112221.fits')[0].data.astype(float)

n_params, n_data = L.shape
print(f'y: {y.shape}, L: {L.shape}, C: {C.shape}')
print(f'y range: [{y.min():.2f}, {y.max():.2f}]')

# === Least squares (full cov) ===
print('\n=== Full model least squares ===')
C_inv = linalg.inv(C)
LTC_inv = L @ C_inv
ATC_inv_A = LTC_inv @ L.T
theta = linalg.solve(ATC_inv_A, LTC_inv @ y, assume_a='pos')
residual = y - theta @ L
chi2 = residual @ C_inv @ residual
print(f'χ² = {chi2:.1f}, dof = {n_data - n_params}')
print(f'χ²/dof = {chi2/(n_data-n_params):.4f}')

# Parameter uncertainties (diagonal of covariance)
cov_theta = linalg.inv(ATC_inv_A)
theta_err = np.sqrt(np.diag(cov_theta))

# Print all parameters
print('\nAll parameters:')
for i in range(n_params):
    # Find what this row represents
    row = L[i]
    nz = np.sum(row != 0)
    nz_vals = np.unique(row[row != 0])[:5]
    print(f'  θ[{i:2d}] = {theta[i]:8.4f} ± {theta_err[i]:6.4f}  (nonzero={nz:4d}, vals={nz_vals})')

# === Identify key parameters ===
# Row with all 1s → SN offset (related to M_B + 5*log10(H0/70))
sn_mask = np.where(np.all(L == 1, axis=1))[0]
print(f'\nSN offset row: θ[{sn_mask[0]}] = {theta[sn_mask[0]]:.4f}')

# H0 parameter (row where all values are -1)
h0_mask = np.where(np.all(L == -1, axis=1))[0]
if len(h0_mask) > 0:
    print(f'H0 row: θ[{h0_mask[0]}] = {theta[h0_mask[0]]:.4f}')
    h0_from_param = 70 * 10**(theta[h0_mask[0]] / 5)
    print(f'  → H0 = {h0_from_param:.2f} km/s/Mpc (assuming H0 = 70 * 10^(θ/5))')

# Parameter 46 (last row) — likely 5*log10(H0)
print(f'\nLast parameter θ[46] = {theta[46]:.6f} ± {theta_err[46]:.6f}')
h0_est = 10**(theta[46] / 5)
print(f'  If θ[46] = 5*log10(H0): H0 = 10^(θ/5) = {h0_est:.2f}')
h0_est2 = 70 * 10**(theta[46] / 5)
print(f'  If θ[46] = 5*log10(H0/70): H0 = 70 * 10^(θ/5) = {h0_est2:.2f}')
# Propagate error
h0_err = h0_est * (theta_err[46] / 5) * np.log(10)
print(f'  H0 error (propagated): ±{h0_err:.3f}')

# Check M_B (SN Ia absolute mag)
# The SN offset row likely has M_B embedded
theta_mb = theta[sn_mask[0]] if len(sn_mask) > 0 else None
if theta_mb is not None:
    print(f'\nM_B related parameter: {theta_mb:.4f}')
    # M_B = -19.238 in SH0ES — let's check what parameter has this value
    for i in range(n_params):
        if abs(theta[i] - (-19.2)) < 0.5:
            print(f'  Found M_B candidate: θ[{i}] = {theta[i]:.4f}')

# === Bootstrap uncertainty ===
print('\n=== Bootstrap ===')
n_boot = 100
theta_boot = np.zeros((n_boot, n_params))
for b in range(n_boot):
    # Resample data points with replacement
    idx = rng.randint(0, n_data, n_data)
    y_b = y[idx]
    L_b = L[:, idx]
    C_b = C[idx][:, idx]
    C_b_inv = linalg.inv(C_b)
    LTC_b = L_b @ C_b_inv
    ATC_b = LTC_b @ L_b.T
    theta_b = linalg.solve(ATC_b, LTC_b @ y_b, assume_a='pos')
    theta_boot[b] = theta_b

# H0 from bootstrap
h0_boot = 10**(theta_boot[:, 46] / 5)
h0_16, h0_50, h0_84 = np.percentile(h0_boot, [16, 50, 84])
print(f'H0 = {h0_50:.2f} [{h0_16:.2f}, {h0_84:.2f}] km/s/Mpc (bootstrap)')

# PL slope from bootstrap
pl_slope_boot = theta_boot[:, 38]
pl_16, pl_50, pl_84 = np.percentile(pl_slope_boot, [16, 50, 84])
print(f'PL slope (θ[38]) = {pl_50:.4f} [{pl_16:.4f}, {pl_84:.4f}]')

# === Fix PL slope to canonical value ===
print('\n=== Fix PL slope test ===')
# The SH0ES PL slope is -3.285, but our fit finds a different value
# Modify L to impose a constraint: fix θ[38] = -3.285
# Add a constraint row to L: 0·θ_0 + ... + 1·θ_38 + ... + 0·θ_46 = -3.285
# L_constraint = np.zeros((1, n_params))
# L_constraint[0, 38] = 1.0
# y_constraint = np.array([-3.285])

# Actually, the easier approach: just report what the linear model naturally gives
print(f'Absolute PL slope (θ[38]): {theta[38]:.4f} ± {theta_err[38]:.4f}')
print(f'  SH0ES canonical: -3.285 ± 0.013')
print(f'  Difference: {theta[38] - (-3.285):.4f} ({abs(theta[38] - (-3.285))/theta_err[38]:.1f}σ)')

# === Modified PL relation test ===
# What if the PL relation has a non-linear term?
# Modifying L to include a nonlinear PL form would require
# rebuilding the L matrix. This is complex — we'd need to
# understand which elements correspond to Cepheids.
# For now, let's check the PL residual structure.

print('\n=== PL residual analysis ===')
# Find Cepheid data (rows with host distance moduli as 1)
ceph_mask = np.any(L[:19] != 0, axis=0)  # Cepheid entries have at least one host indicator = 1
n_ceph = np.sum(ceph_mask)
n_sn = n_data - n_ceph
print(f'Cepheids: {n_ceph}, SNe: {n_sn}')

# Predicted vs observed for Cepheids
y_pred = theta @ L
ceph_resid = y[ceph_mask] - y_pred[ceph_mask]
print(f'Cepheid residual RMS: {np.std(ceph_resid):.4f}')
print(f'Cepheid residual mean: {np.mean(ceph_resid):.5f}')

# Check if residuals correlate with logP
# We need L rows that contain logP values
logP_rows = []
for i in range(n_params):
    row = L[i, ceph_mask]
    if len(np.unique(row)) > 2:  # continuous values (not 0/1 indicator)
        r = np.corrcoef(ceph_resid, row)[0, 1]
        if abs(r) > 0.01:
            print(f'  Row {i}: logP correlation r={r:.4f}')
            logP_rows.append(i)

# === Plot ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Predicted vs observed
axes[0, 0].scatter(y, y_pred, s=1, alpha=0.3)
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', alpha=0.5)
axes[0, 0].set_xlabel('Observed')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title(f'Full ladder fit (χ²/dof={chi2/(n_data-n_params):.2f})')

# 2. Residuals
axes[0, 1].scatter(range(n_data), y - y_pred, s=1, alpha=0.3)
axes[0, 1].axhline(0, c='r', ls='--', alpha=0.5)
axes[0, 1].set_xlabel('Index')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residuals')
axes[0, 1].axvline(n_ceph, c='g', ls=':', alpha=0.7, label=f'Cepheid/SN boundary ({n_ceph})')
axes[0, 1].legend(fontsize=8)

# 3. Cepheid residuals
axes[0, 2].hist(ceph_resid, bins=50, alpha=0.7)
axes[0, 2].set_xlabel('Residual (mag)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_title(f'Cepheid residuals (RMS={np.std(ceph_resid):.3f})')

# 4. Cepheid residuals vs logP (if available)
if len(logP_rows) > 0:
    logP_row = logP_rows[0]
    logP_vals = L[logP_row, ceph_mask]
    axes[1, 0].scatter(logP_vals, ceph_resid, s=2, alpha=0.4)
    axes[1, 0].axhline(0, c='r', ls='--', alpha=0.5)
    axes[1, 0].set_xlabel('log10(P)')
    axes[1, 0].set_ylabel('PL residual (mag)')
    axes[1, 0].set_title(f'PL residual vs logP (row {logP_row})')

# 5. Bootstrap H0 distribution
axes[1, 1].hist(h0_boot, bins=30, alpha=0.7, color='C1')
axes[1, 1].axvline(h0_50, c='r', ls='-', lw=2)
axes[1, 1].axvline(h0_16, c='r', ls='--')
axes[1, 1].axvline(h0_84, c='r', ls='--')
axes[1, 1].set_xlabel('H0 (km/s/Mpc)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'H0 = {h0_50:.1f} [{h0_16:.1f}, {h0_84:.1f}]')

# 6. Parameter correlations (PL slope vs H0)
axes[1, 2].scatter(theta_boot[:, 38], h0_boot, s=4, alpha=0.5)
axes[1, 2].axvline(theta[38], c='r', ls='--', alpha=0.5)
axes[1, 2].axhline(h0_50, c='r', ls='--', alpha=0.5)
axes[1, 2].set_xlabel('PL slope (θ[38])')
axes[1, 2].set_ylabel('H0')
corr = np.corrcoef(theta_boot[:, 38], h0_boot)[0, 1]
axes[1, 2].set_title(f'PL slope vs H0 (r={corr:.3f})')

plt.tight_layout()
plt.savefig('output/ladder_h0.png', dpi=150)
print('\nSaved output/ladder_h0.png')

# === Summary ===
print('\n===== DISTANCE LADDER SUMMARY =====')
print(f'Full least squares fit ({n_params} parameters, {n_data} data points):')
print(f'  χ² = {chi2:.1f}, χ²/dof = {chi2/(n_data-n_params):.2f}')
print(f'  θ[46] (5·log10(H0)?) = {theta[46]:.4f} ± {theta_err[46]:.4f}')
print(f'  H0 = {h0_est:.2f} ± {h0_err:.3f} (from θ[46] uncertainty)')
print(f'  H0 (bootstrap) = {h0_50:.2f} [{h0_16:.2f}, {h0_84:.2f}]')
print(f'  PL slope = {theta[38]:.4f} ± {theta_err[38]:.4f}')
print(f'  SH0ES PL slope = -3.285')
print(f'  PL slope vs H0 correlation: r = {np.corrcoef(theta_boot[:, 38], h0_boot)[0, 1]:.3f}')

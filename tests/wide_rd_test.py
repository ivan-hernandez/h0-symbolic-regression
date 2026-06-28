"""Test: fit with r_s as free parameter, uniform prior [130, 160] Mpc.
Uses raw D_H/r_s and D_M/r_s from DESI DR2.
"""
import sys, os, math, numpy as np
from scipy import optimize, interpolate
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as D

C = 299792.458

# Load CC, SNe
cc = D.get_cc(); z_cc, h_cc, e_cc = cc[:,0], cc[:,1], cc[:,2]
z_s, mu_s, e_s = D.fetch_pantheon()

# SDSS BAO: D_V/r_s
# [z, D_V/r_s (Mpc), error]
bao_sdss = np.array([
    [0.38, 1509, 24],
    [0.51, 1976, 27],
    [0.61, 2297, 35],
])

# DESI DR2 raw: [z, D_H/r_s, D_M/r_s, cov(H,H), cov(H,M), cov(M,M)]
# From arXiv:2503.14738
desi_cov = np.array([
    # z=0.51
    [0.510, 93.28, 13.25, 0.184, 0.0310, 0.502],
    # z=0.706
    [0.706, 104.82, 15.59, 0.111, 0.0180, 0.656],
    # z=0.934
    [0.934, 115.60, 18.51, 0.0404, 0.00715, 0.253],
    # z=1.321
    [1.321, np.nan, 20.85, 0, 0, 0.375],
    # z=1.484
    [1.484, np.nan, 21.76, 0, 0, 0.895],
    # z=2.330
    [2.330, 8.63, 26.97, 0.0102, 0.00754, 0.639],
])

# Model: Cpx 13 for H(z)
def H_model(z, H0, A, B, Cv):
    return H0 + A * z * (z - B) * (z*z + Cv)

def get_distance_integrals(z_arr, H0, A, B, Cv):
    """Compute D_c(z) = c*integral dz'/H(z') for given redshifts."""
    z_max = max(np.max(z_arr), 0.001)
    n_grid = max(200, int(z_max * 200))
    grid = np.linspace(0, z_max, n_grid)
    inv_H = 1.0 / np.maximum(H0 + A * grid * (grid - B) * (grid*grid + Cv), 1e-10)
    dz = grid[1] - grid[0]
    cum = np.zeros(n_grid)
    cum[1:] = np.cumsum(dz * (inv_H[:-1] + inv_H[1:]) / 2)
    f = interpolate.interp1d(grid, cum * C, bounds_error=False, fill_value=0.0)
    Dc = f(z_arr)
    return Dc

def mu_batch(z_arr, H0, A, B, Cv):
    Dc = get_distance_integrals(z_arr, H0, A, B, Cv)
    return 5.0 * np.log10(np.maximum((1 + z_arr) * Dc, 1e-10)) + 25.0

def predict_bao(z, H0, A, B, Cv, rs):
    """Predict D_V/r_s, D_H/r_s, D_M/r_s."""
    Hz = H_model(z, H0, A, B, Cv)
    Dc = get_distance_integrals(np.array([z]), H0, A, B, Cv)[0]
    DA = Dc / (1 + z) if z > 0 else Dc
    D_H_rs = C / (Hz * rs)
    D_M_rs = Dc / rs
    # D_V = (c*z/H * ((1+z)*D_A)^2)^(1/3) = (c*z*Dc^2 / H(z))^(1/3)
    D_V = (C * z * Dc**2 / Hz)**(1/3)
    D_V_rs = D_V / rs
    return D_H_rs, D_M_rs, D_V_rs

def chi2_total(params):
    H0, A, B, Cv, rs = params
    if not (50 < H0 < 80) or not (130 < rs < 160):
        return 1e10
    
    # CC
    pred = H0 + A * z_cc * (z_cc - B) * (z_cc*z_cc + Cv)
    chi2 = np.sum(((h_cc - pred) / e_cc)**2)
    
    # SDSS BAO (D_V/r_s)
    for z, dv_rs, err in bao_sdss:
        _, _, pred_dv_rs = predict_bao(z, H0, A, B, Cv, rs)
        chi2 += ((dv_rs - pred_dv_rs) / err)**2
    
    # DESI DR2 (D_H/r_s and D_M/r_s with covariance)
    for row in desi_cov:
        z = row[0]
        dh_obs = row[1]; dm_obs = row[2]
        var_hh = row[3]; cov_hm = row[4]; var_mm = row[5]
        if np.isnan(dh_obs):
            # D_M only for high-z
            pred_dh, pred_dm, _ = predict_bao(z, H0, A, B, Cv, rs)
            chi2 += ((dm_obs - pred_dm)**2 / var_mm)
        else:
            pred_dh, pred_dm, _ = predict_bao(z, H0, A, B, Cv, rs)
            dx = np.array([dh_obs - pred_dh, dm_obs - pred_dm])
            cov = np.array([[var_hh, cov_hm], [cov_hm, var_mm]])
            det = var_hh * var_mm - cov_hm**2
            if det > 0:
                icov = np.array([[var_mm, -cov_hm], [-cov_hm, var_hh]]) / det
                chi2 += dx @ icov @ dx
    
    # SNe
    pred_mu = mu_batch(z_s, H0, A, B, Cv)
    resid = mu_s - pred_mu
    suma = np.sum(resid / e_s**2)
    chi2 += np.sum(((resid - suma/np.sum(1.0/e_s**2)) / e_s)**2)
    
    # Weak H0 prior
    chi2 += ((H0 - 67.4) / 20.0)**2
    
    return chi2

# Fit with fixed r_s=147
print("Fitting with r_s=147 (fixed)...")
res_fixed = optimize.minimize(
    lambda p: chi2_total([p[0], p[1], p[2], p[3], 147.0]),
    [68, -7, 3.8, 1.7], method='L-BFGS-B',
    bounds=[(50,80), (-15,0), (0,8), (-2,5)],
    options={'maxiter':5000})
print(f"  H0={res_fixed.x[0]:.2f} A={res_fixed.x[1]:.2f} B={res_fixed.x[2]:.2f} C={res_fixed.x[3]:.2f}")
print(f"  chi2={res_fixed.fun:.1f}")

# Fit with r_s free
print("\nFitting with r_s free [130, 160]...")
res_free = optimize.minimize(
    chi2_total,
    [68, -7, 3.8, 1.7, 147],
    method='L-BFGS-B',
    bounds=[(50,80), (-15,0), (0,8), (-2,5), (130,160)],
    options={'maxiter':5000})
print(f"  H0={res_free.x[0]:.2f} A={res_free.x[1]:.2f} B={res_free.x[2]:.2f} C={res_free.x[3]:.2f}")
print(f"  r_s={res_free.x[4]:.1f}")
print(f"  chi2={res_free.fun:.1f}")
print(f"  ΔH0 = {res_free.x[0] - res_fixed.x[0]:.2f} (free r_s - fixed 147)")

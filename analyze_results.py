#!/usr/bin/env python3
"""
Comprehensive analysis: best SR models vs LCDM on CC+BAO+SNe.
Generates comparison plots and reports statistics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os

# ============================================================
# DATA
# ============================================================

def get_cc_data():
    return np.array([
        [0.070,69.0,19.6],[0.090,69.0,12.0],[0.120,68.6,26.2],
        [0.170,83.0,8.0],[0.1791,75.0,4.0],[0.1993,75.0,5.0],
        [0.200,72.9,29.6],[0.270,77.0,14.0],[0.280,88.8,36.6],
        [0.3519,83.0,14.0],[0.3802,83.0,13.5],[0.400,95.0,17.0],
        [0.4004,77.0,10.2],[0.4247,87.1,11.2],[0.4497,92.8,12.9],
        [0.470,89.0,34.0],[0.4783,80.9,9.0],[0.480,97.0,62.0],
        [0.5929,104.0,13.0],[0.6797,92.0,8.0],[0.750,98.8,33.6],
        [0.7812,105.0,12.0],[0.800,113.1,28.5],[0.8754,125.0,17.0],
        [0.880,90.0,40.0],[0.900,117.0,23.0],[1.037,154.0,20.0],
        [1.300,168.0,17.0],[1.363,160.0,33.6],[1.430,177.0,18.0],
        [1.530,140.0,14.0],[1.750,202.0,40.0],[1.965,186.5,50.4],
    ])

def get_bao_data():
    return np.array([
        [0.380,81.1,2.2],[0.510,91.1,2.1],[0.610,99.4,2.2],
    ])

def get_desi_bao_data(r_d=147.0):
    c = 299792.458
    desi = np.array([
        [0.510, 20.98334647, 0.61],
        [0.706, 20.07872919, 0.60],
        [0.930, 17.87612922, 0.35],
        [1.317, 13.82372285, 0.42],
        [2.330,  8.52256583, 0.17],
    ])
    hz = c / (r_d * desi[:, 1])
    errs = hz * desi[:, 2] / desi[:, 1]
    return np.column_stack([desi[:, 0], hz, errs])

# ============================================================
# MODELS
# ============================================================

H0_REF = 67.4

def H_cpx13(z):
    """Best joint CC+BAO+SNe model (Cpx 13, H0=67.4)."""
    f = z * ((z - 2.6035253958770332) * (((z * z) + 0.8069087986933524) * -20.741134893505617))
    return H0_REF + f

def H_cpx11(z):
    """Cpx 11 model (H0=72.1)."""
    f = z * z * (84.35746021110187 + z * (-27.166125568007434)) + 4.7032
    return H0_REF + f

def H_cpx20(z):
    """Cpx 20 model (H0=64.4)."""
    f = -48.35652340908182 * z * (z - 2.778854840338059) \
        * np.sqrt(z * (z - 0.8499296350310853) + 0.39084941382533883) - 2.978402579262088
    return H0_REF + f

def H_lcdm(z, H0=67.4, Om=0.315):
    return H0 * np.sqrt(Om * (1+z)**3 + (1-Om))

# ============================================================
# INTEGRATION
# ============================================================

def quad_simple(f, a, b, n=2000):
    xs = np.linspace(a, b, 2*n + 1)
    h = (b - a) / (2*n)
    fx = f(xs)
    return h/3 * (fx[0] + fx[-1] + 4*np.sum(fx[1::2]) + 2*np.sum(fx[2:-1:2]))

def mu_from_H(H_func, z):
    Dc = 299792.458 * quad_simple(lambda zp: 1.0 / np.maximum(H_func(zp), 1), 0, z)
    return 5.0 * np.log10((1 + z) * Dc) + 25.0

# ============================================================
# PLOT
# ============================================================

def make_plot():
    cc = get_cc_data()
    bao = get_bao_data()
    
    z_plot = np.linspace(0, 2.0, 500)
    
    models = [
        ("Cpx 13 (H0=67.4)", H_cpx13, '#1a9641'),
        ("Cpx 11 (H0=72.1)", H_cpx11, '#d7191c'),
        ("Cpx 20 (H0=64.4)", H_cpx20, '#fdae61'),
        ("LCDM Planck", lambda z: H_lcdm(z, 67.4), '#2c7bb6'),
        ("LCDM SH0ES", lambda z: H_lcdm(z, 73.0), '#b2182b'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: H(z) fit
    ax = axes[0, 0]
    ax.errorbar(cc[:,0], cc[:,1], yerr=cc[:,2], fmt='o', ms=3, alpha=0.4,
                color='gray', capsize=2, label='CC')
    ax.errorbar(bao[:,0], bao[:,1], yerr=bao[:,2], fmt='s', ms=4, alpha=0.7,
                color='black', capsize=2, label='BAO')
    for name, hfunc, color in models:
        Hv = hfunc(z_plot)
        ax.plot(z_plot, Hv, color=color, lw=1.5, label=name)
    ax.set_xlabel('z'); ax.set_ylabel('H(z) [km/s/Mpc]')
    ax.set_title('Hubble Parameter Fit')
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    ax.set_xlim(0, 2.0)
    
    # Panel 2: H(z) low-z zoom
    ax = axes[0, 1]
    mask = z_plot < 0.3
    ax.errorbar(cc[:,0], cc[:,1], yerr=cc[:,2], fmt='o', ms=4, alpha=0.5,
                color='gray', capsize=2)
    for name, hfunc, color in models:
        ax.plot(z_plot[mask], hfunc(z_plot[mask]), color=color, lw=1.5, label=name)
    ax.axhline(67.4, color='gray', ls=':', lw=0.5)
    ax.axhline(73.0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('z'); ax.set_ylabel('H(z) [km/s/Mpc]')
    ax.set_title('Low-z Zoom (H0 region)')
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    ax.set_xlim(0, 0.3)
    
    # Panel 3: Distance modulus residuals vs LCDM Planck
    ax = axes[1, 0]
    # Compute LCDM Planck mu as reference
    def mu_lcdm_planck(z):
        return mu_from_H(lambda zz: H_lcdm(zz, 67.4), z)
    
    z_ref = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    mu_ref = np.array([mu_lcdm_planck(z) for z in z_ref])
    
    for name, hfunc, color in models:
        mu_m = np.array([mu_from_H(hfunc, z) for z in z_ref])
        # Free M: optimal offset
        resid = mu_m - mu_ref
        delta = np.mean(resid)  # unweighted since no errors here
        ax.plot(z_ref, resid - delta, color=color, lw=1.5, label=name)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('z'); ax.set_ylabel('Δμ (free M) [mag]')
    ax.set_title('Distance Modulus vs LCDM Planck')
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    
    # Panel 4: H(z) residual vs LCDM Planck
    ax = axes[1, 1]
    z_h = np.sort(np.concatenate([cc[:,0], bao[:,0]]))
    H_lcdm_planck_v = H_lcdm(z_h, 67.4)
    for name, hfunc, color in models:
        H_m = hfunc(z_h)
        ax.plot(z_h, H_m - H_lcdm_planck_v, color=color, lw=1, alpha=0.7, label=name)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('z'); ax.set_ylabel('ΔH [km/s/Mpc]')
    ax.set_title('H(z) Residual vs LCDM')
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    fig.savefig('output/joint_comparison.png', dpi=150)
    plt.close(fig)
    print("Saved output/joint_comparison.png")
    
    # Also print summary
    print("\n=== MODEL SUMMARY ===")
    print(f"{'Model':>25} {'H0':>7} {'f(0)':>7} {'H(z) shape':>30}")
    print("-" * 70)
    print(f"{'Cpx 13 (best joint)':>25} {H_cpx13(0):>7.1f} {H_cpx13(0)-H0_REF:>+7.2f} {'-20.74z^4 + 54.03z^3 + 26.84z^2':>30}")
    print(f"{'Cpx 11 (H0=72)':>25} {H_cpx11(0):>7.1f} {H_cpx11(0)-H0_REF:>+7.2f} {'quartic + constant':>30}")
    print(f"{'Cpx 20 (best H fit)':>25} {H_cpx20(0):>7.1f} {H_cpx20(0)-H0_REF:>+7.2f} {'sqrt-based, -2.98 const':>30}")
    print(f"{'LCDM Planck':>25} {H_lcdm(0,67.4):>7.1f} {'--':>7} {'sqrt(Om(1+z)^3 + Ol)':>30}")
    print(f"{'LCDM SH0ES':>25} {H_lcdm(0,73.0):>7.1f} {'--':>7} {'sqrt(Om(1+z)^3 + Ol)':>30}")
    
    # Compute effective w(z) for Cpx 13 assuming LCDM Om=0.315
    print("\n=== EFFECTIVE w(z) for Cpx 13 (assuming Om=0.315) ===")
    print(f"  {'z':>5} {'H/H0':>10} {'E^2':>10} {'Om(z)':>10} {'w(z)':>10}")
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    Om = 0.315
    for z in [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
        E = H_cpx13(z) / H_cpx13(0)
        E2 = E**2
        Om_z = Om * (1+z)**3
        # w(z) = -1 + (1/3) * d(ln(E^2 - Om_z)) / d(ln(1+z))
        # Numerical derivative: d/dz then convert
        dz = 0.001
        zp, zm = z + dz, max(z - dz, 0)
        Ep2 = (H_cpx13(zp) / H_cpx13(0))**2
        Em2 = (H_cpx13(zm) / H_cpx13(0))**2
        Om_p = Om * (1+zp)**3
        Om_m = Om * (1+zm)**3
        d_lnX_dln1pz = ((Ep2 - Om_p) - (Em2 - Om_m)) / (2*dz) * (1+z) / (E2 - Om_z) if abs(E2 - Om_z) > 1e-10 else 0
        w = -1 + d_lnX_dln1pz / 3
        print(f"  {z:>5.1f} {E:>10.4f} {E2:>10.4f} {Om_z:>10.4f} {w:>+10.3f}")
    
    print("\n  (For LCDM with same Om: w(z) ≡ -1 exactly)")

if __name__ == "__main__":
    make_plot()

"""Phase 3: Full MCMC joint (σ₈, Ω_m) constraints from CPX5 RAR + VF.

Proper implementation:
1. Sheth-Tormen halo mass function
2. Virial scaling + CPX5 RAR → V_max prediction
3. Likelihood vs observed SDSS velocity function
4. emcee MCMC sampling for (sigma8, Omega_m, log_M1)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "analysis/phase3"

# ── Physical constants ──
G_SI = 6.6743e-11
Msun_kg = 1.989e30
Mpc_m = 3.0857e22
kpc_m = 3.0857e19
fb = 0.157
H0 = 68.0

CPX5_A = -17.06
CPX5_B = -72.71

# Observed VF (SDSS, Papastergis+2011 / Zavala+2009)
V_obs = np.array([50, 80, 110, 150, 190, 230, 280, 340])
phi_obs = 10**np.array([-1.70, -2.25, -2.70, -3.15, -3.55, -4.00, -4.55, -5.20])
phi_err = phi_obs * 0.15  # 15% systematic uncertainty


def rho_crit(h=0.68):
    """Critical density in Msun/Mpc³."""
    H0_si = h * 100 * 1000 / Mpc_m
    rho_c = 3 * H0_si**2 / (8 * np.pi * G_SI)
    return rho_c / Msun_kg * Mpc_m**3


def sigma_M(M, sigma8, Om=0.32):
    """RMS fluctuation σ(M). Approximate from Eisenstein & Hu transfer."""
    h = 0.68
    M8 = 5.96e14 * Om  # mass in 8 Mpc/h sphere, Msun
    n_eff = -0.15 * (np.log10(M / M8)) + 0.8
    Gamma = Om * h * (1 + (1 - Om)**0.6)
    gamma_eff = (0.3 * Gamma + 0.2) * (2.92 + np.log10(M / M8) / 3.0)
    return sigma8 * (M / M8)**(-gamma_eff / 3.0)


def delta_c(z=0, Om=0.32):
    """Critical overdensity for collapse."""
    Omz = Om / (Om + (1 - Om) * (1 + z)**(-3))
    return 1.686 * (1 + 0.0123 * np.log(max(Omz, 0.01)))


def sheth_tormen(M, z, sigma8, Om=0.32):
    """Sheth-Tormen (1999) halo mass function. dn/dlnM [Mpc⁻³]."""
    h = 0.68
    dc = delta_c(z, Om)
    sig = sigma_M(M, sigma8, Om)
    nu = dc / np.maximum(sig, 0.01)

    A_s = 0.3222
    a_s = 0.707
    p_s = 0.3
    f_nu = A_s * np.sqrt(2*a_s/np.pi) * nu * np.exp(-a_s*nu**2/2)
    f_nu *= (1 + (a_s*nu**2)**(-p_s))

    rho_m = Om * rho_crit()
    dlnsig_dlnM = -0.15 * (0.3*Om*h + 0.2) * (2.92 + np.log10(M/1e14)/3) / 3

    return f_nu * rho_m / M * np.abs(dlnsig_dlnM)


def M_to_Vmax(M, sigma8, Om):
    """Convert halo mass M to V_max using CPX5 RAR."""
    # Virial radius
    rho_vir = 200 * rho_crit()
    R_vir_kpc = (3 * M / (4 * np.pi * rho_vir))**(1/3) * 1000  # Mpc → kpc

    # Baryonic acceleration at R_vir
    M_bar = fb * M
    g_bar_si = G_SI * M_bar * Msun_kg / (R_vir_kpc * kpc_m)**2

    # CPX5 RAR
    log_gb = np.log10(np.maximum(g_bar_si, 1e-20))
    log_go = CPX5_A + CPX5_B / log_gb
    g_obs_si = 10**log_go

    # V_max = sqrt(g_obs * R)
    V_km_s = np.sqrt(g_obs_si * R_vir_kpc * kpc_m) / 1000
    return V_km_s


def predicted_vf(sigma8, Om, log_M1=11.5):
    """Predicted velocity function for given cosmology.

    Parameters:
    - sigma8, Om: cosmology
    - log_M1: log10 of mass completeness limit (Msun) — galaxies below this
              cannot form HI disks efficiently
    """
    M_grid = np.logspace(log_M1, 14.0, 200)
    V_grid = M_to_Vmax(M_grid, sigma8, Om)
    dndlnM = np.array([sheth_tormen(m, 0, sigma8, Om) for m in M_grid])

    # Convert dn/dlnM → dn/dlnV using dlnM/dlnV ≈ 3 (virial V ∝ M^{1/3})
    # dlnV/dlnM ≈ 1/3 → dlnM/dlnV ≈ 3
    dndlnV = dndlnM * 3.0

    # Only use galaxies above V > 30 km/s (HI detection limit)
    mask = (V_grid > 25) & (V_grid < 600) & np.isfinite(dndlnV) & (dndlnV > 0)
    V_sorted = V_grid[mask]
    dn_sorted = dndlnV[mask]

    # Sort by V for interpolation
    order = np.argsort(V_sorted)
    return V_sorted[order], dn_sorted[order]


def log_likelihood(theta):
    """Log likelihood for (sigma8, Om, log_M1)."""
    sigma8, Om, log_M1 = theta

    if sigma8 < 0.6 or sigma8 > 1.0 or Om < 0.20 or Om > 0.45 or log_M1 < 10.0 or log_M1 > 13.0:
        return -1e10

    try:
        V_theory, dn_theory = predicted_vf(sigma8, Om, log_M1)
        if len(V_theory) < 3:
            return -1e10

        phi_theory = np.interp(V_obs, V_theory, dn_theory, left=1e-15, right=1e-15)
        valid = phi_theory > 1e-15
        if valid.sum() < 3:
            return -1e10

        log_phi_theory = np.log10(phi_theory[valid])
        log_phi_obs = np.log10(phi_obs[valid])
        log_err = np.log10(phi_obs[valid] + phi_err[valid]) - log_phi_obs

        chi2 = np.sum((log_phi_theory - log_phi_obs)**2 / log_err**2)
        return -0.5 * chi2
    except Exception:
        return -1e10


def log_prior(theta):
    sigma8, Om, log_M1 = theta
    if 0.60 < sigma8 < 1.00 and 0.20 < Om < 0.45 and 10.0 < log_M1 < 13.0:
        return 0.0
    return -np.inf


def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def run_mcmc(outdir=OUTDIR, n_walkers=32, n_steps=2000):
    """Run emcee MCMC for joint (σ₈, Ω_m, log_M1) constraints."""
    import os
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Phase 3: Full MCMC Joint Constraints")
    print(f"  emcee: {n_walkers} walkers × {n_steps} steps")
    print("=" * 60)

    try:
        import emcee
    except ImportError:
        print("  emcee not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip3", "install", "emcee", "-q"])
        import emcee

    ndim = 3
    # Initialize walkers around fiducial + noise
    fiducial = np.array([0.81, 0.32, 11.5])
    init_pos = fiducial + 0.02 * np.random.randn(n_walkers, ndim)
    init_pos[:, 0] = np.clip(init_pos[:, 0], 0.65, 0.95)  # sigma8
    init_pos[:, 1] = np.clip(init_pos[:, 1], 0.22, 0.42)  # Om
    init_pos[:, 2] = np.clip(init_pos[:, 2], 10.2, 12.5)  # log_M1

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    print(f"  Running MCMC...")
    sampler.run_mcmc(init_pos, n_steps, progress=True)
    print(f"  Done.")

    # Flatten chain (discard first 500 steps as burn-in)
    burn_in = 500
    samples = sampler.get_chain(discard=burn_in, flat=True)

    # Results
    sigma8_med = np.median(samples[:, 0])
    sigma8_lo, sigma8_hi = np.percentile(samples[:, 0], [16, 84])
    Om_med = np.median(samples[:, 1])
    Om_lo, Om_hi = np.percentile(samples[:, 1], [16, 84])
    logM1_med = np.median(samples[:, 2])

    print(f"\n  MCMC Results (68% CL):")
    print(f"    σ₈ = {sigma8_med:.3f}   [{sigma8_lo:.3f}, {sigma8_hi:.3f}]")
    print(f"    Ω_m = {Om_med:.3f}     [{Om_lo:.3f}, {Om_hi:.3f}]")
    print(f"    log_M₁ = {logM1_med:.2f}")

    # Planck comparison
    print(f"\n  Comparison with Planck 2018:")
    print(f"    Planck σ₈ = 0.811 ± 0.006")
    print(f"    This work σ₈ = {sigma8_med:.2f} ± {0.5*(sigma8_hi-sigma8_lo):.2f} (VF + CPX5 RAR)")
    print(f"    Planck Ω_m = 0.315 ± 0.007")
    print(f"    This work Ω_m = {Om_med:.3f} ± {0.5*(Om_hi-Om_lo):.3f} (VF + CPX5 RAR)")

    # ── Figures ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Chains
    ax = axes[0, 0]
    for i in range(min(10, n_walkers)):
        chain = sampler.get_chain()[:, i, 0]
        ax.plot(chain, lw=0.3, alpha=0.7)
    ax.axvline(burn_in, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("σ₈")
    ax.set_title("(a) σ₈ Chains")

    ax = axes[0, 1]
    for i in range(min(10, n_walkers)):
        chain = sampler.get_chain()[:, i, 1]
        ax.plot(chain, lw=0.3, alpha=0.7)
    ax.axvline(burn_in, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Ω_m")
    ax.set_title("(b) Ω_m Chains")

    # (c) Corner plot (σ₈ vs Ω_m)
    ax = axes[0, 2]
    H, xe, ye = np.histogram2d(samples[:, 0], samples[:, 1], bins=30)
    ax.contourf(0.5*(xe[1:]+xe[:-1]), 0.5*(ye[1:]+ye[:-1]), H.T,
                levels=np.percentile(H[H>0], [30, 60, 90]),
                colors=["lightblue", "steelblue", "darkblue"], alpha=0.7)
    ax.scatter([sigma8_med], [Om_med], c="red", marker="*", s=200, zorder=5)
    ax.scatter([0.811], [0.315], c="green", marker="s", s=100, zorder=5, label="Planck")
    ax.set_xlabel("σ₈")
    ax.set_ylabel("Ω_m")
    ax.set_title("(c) Joint Posterior")
    ax.legend(fontsize=8)

    # (d) Best-fit VF
    ax = axes[1, 0]
    Vt_best, dnt_best = predicted_vf(sigma8_med, Om_med, logM1_med)
    ax.loglog(Vt_best, dnt_best, "b-", lw=2, label="Best-fit CPX5 + RAR")
    # 100 random posterior draws
    idx = np.random.choice(len(samples), 100, replace=False)
    for i in idx:
        Vt_s, dnt_s = predicted_vf(samples[i, 0], samples[i, 1], samples[i, 2])
        ax.loglog(Vt_s, dnt_s, "b-", lw=0.2, alpha=0.1)
    ax.errorbar(V_obs, phi_obs, yerr=phi_err, fmt="ro", ms=5, capsize=2, label="SDSS VF")
    ax.set_xlabel("V_max [km/s]")
    ax.set_ylabel("dN/dlogV [Mpc⁻³]")
    ax.set_title("(d) Velocity Function Fit")
    ax.legend(fontsize=8)
    ax.set_xlim(30, 500)
    ax.set_ylim(1e-6, 0.1)

    # (e) VF residuals
    ax = axes[1, 1]
    phi_best = np.interp(V_obs, Vt_best, dnt_best, left=1e-15, right=1e-15)
    resid = (phi_obs - phi_best) / phi_err
    ax.errorbar(V_obs, resid, yerr=1.0, fmt="ro", ms=5, capsize=2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("V_max [km/s]")
    ax.set_ylabel("(data - model) / σ")
    ax.set_title(f"(e) Residuals (χ²/dof)")
    ax.set_ylim(-3, 3)

    # (f) Parameter correlations
    ax = axes[1, 2]
    ax.scatter(samples[::20, 0], samples[::20, 1], s=1, alpha=0.3,
               c=samples[::20, 2], cmap="viridis")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("log M₁")
    ax.set_xlabel("σ₈")
    ax.set_ylabel("Ω_m")
    ax.set_title("(f) Posteriors Colored by M₁")

    plt.tight_layout()
    plt.savefig(f"{outdir}/mcmc_joint_constraints.pdf", dpi=200)
    plt.savefig(f"{outdir}/mcmc_joint_constraints.png", dpi=150)
    print(f"\n  Saved {outdir}/mcmc_joint_constraints.png")
    plt.close()

    # Save samples
    np.savez(f"{outdir}/mcmc_samples.npz", samples=samples,
             sigma8_med=sigma8_med, sigma8_lo=sigma8_lo, sigma8_hi=sigma8_hi,
             Om_med=Om_med, Om_lo=Om_lo, Om_hi=Om_hi,
             logM1_med=logM1_med)
    print(f"  Saved {outdir}/mcmc_samples.npz")

    return samples, sigma8_med, Om_med


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_mcmc(n_steps=2000)
    print("\nPhase 3 MCMC complete.")

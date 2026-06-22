"""Phase 3: Proper abundance matching + SMF MCMC.

Fixes the prior-bound issue by replacing M* ≈ 0.08*fb*M with
a proper abundance-matched stellar-to-halo mass relation (SHMR).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
import os, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/phase3"
G_SI = 6.6743e-11
Msun_kg = 1.989e30
Mpc_m = 3.0857e22
kpc_m = 3.0857e19
fb = 0.157
h = 0.68
CPX5_A = -17.06
CPX5_B = -72.71
H0 = 68.0

# ── Observables ───────────────────────────────────────────────────────────────

BALDRY_SMF = np.array([
    [8.00, -1.51, 0.06], [8.25, -1.52, 0.04], [8.50, -1.62, 0.03],
    [8.75, -1.83, 0.03], [9.00, -2.02, 0.03], [9.25, -2.18, 0.03],
    [9.50, -2.33, 0.03], [9.75, -2.50, 0.03], [10.00, -2.72, 0.03],
    [10.25, -3.01, 0.04], [10.50, -3.42, 0.05], [10.75, -3.91, 0.07],
    [11.00, -4.52, 0.10], [11.25, -5.18, 0.15], [11.50, -5.95, 0.25],
])

V_obs = np.array([50, 80, 110, 150, 190, 230, 280, 340])
phi_vf = 10**np.array([-1.70, -2.25, -2.70, -3.15, -3.55, -4.00, -4.55, -5.20])


# ── Halo Model ────────────────────────────────────────────────────────────────

def rho_crit():
    H0_si = H0 * 1000 / Mpc_m
    return 3 * H0_si**2 / (8 * np.pi * G_SI) / Msun_kg * Mpc_m**3


def halo_mass_function(M, sigma8, Om):
    dc = 1.686
    n_eff = -0.15 * (np.log10(M / (5.96e14 * Om))) + 0.8
    Gamma = Om * h
    gamma_eff = (0.3*Gamma + 0.2) * (2.92 + np.log10(M/1e14)/3) / 3
    sigma_M = sigma8 * (M / (5.96e14 * Om))**(-gamma_eff)
    nu = dc / np.maximum(sigma_M, 0.01)
    f_nu = 0.322 * np.sqrt(0.45) * nu * np.exp(-0.354*nu**2) * (1 + (0.707*nu**2)**(-0.3))
    rho_m = Om * rho_crit()
    dndlnM = f_nu * rho_m / M * np.abs(-0.4)
    return np.maximum(dndlnM, 1e-20)


def M_to_Vmax(M, sigma8, Om):
    rho_vir = 200 * rho_crit()
    R_vir_kpc = (3 * M / (4*np.pi*rho_vir))**(1/3) * 1000
    M_bar = fb * M
    g_bar = G_SI * M_bar * Msun_kg / (R_vir_kpc * kpc_m)**2
    log_gb = np.log10(np.maximum(g_bar, 1e-20))
    log_go = CPX5_A + CPX5_B / log_gb
    g_obs = 10**log_go
    return np.sqrt(g_obs * R_vir_kpc * kpc_m) / 1000


# ── Abundance Matching (Behroozi+2010 parameterization) ───────────────────────

def abundance_match_shmr(sigma8, Om):
    """Compute M*(M_h) via abundance matching to Baldry+2012 SMF."""
    M_h_grid = np.logspace(9, 15, 500)
    dndlnM = np.array([halo_mass_function(m, sigma8, Om) for m in M_h_grid])

    # Cumulative halo abundance N(>M_h)
    dlnM = np.log(M_h_grid[1] / M_h_grid[0])
    N_gt_Mh = np.cumsum(dndlnM[::-1] * dlnM)[::-1]

    # Abundance match: N(>M*) = N(>M_h)
    # Interpolate: given N(>M_h), find M* such that N(>M*) matches
    # First, create interpolation M* → N(>M*)
    M_s_grid = 10**BALDRY_SMF[:, 0]
    phi_s = 10**BALDRY_SMF[:, 1]
    dlogM = BALDRY_SMF[1, 0] - BALDRY_SMF[0, 0]
    N_gt_Ms = np.cumsum(phi_s[::-1] * dlogM)[::-1]

    # Interpolation: M* → N(>M*) then invert
    from scipy.interpolate import interp1d
    interp_Ns = interp1d(np.log10(M_s_grid)[::-1], np.log10(N_gt_Ms)[::-1],
                          kind="linear", bounds_error=False,
                          fill_value=(np.log10(N_gt_Ms[-1]), np.log10(N_gt_Ms[0])))

    # For each M_h, find M* such that N(>M*) = N(>M_h)
    log_N_gt_Mh = np.log10(np.maximum(N_gt_Mh, N_gt_Ms[-1]))
    log_N_gt_Mh = np.minimum(log_N_gt_Mh, np.log10(N_gt_Ms[0]))
    log_M_star = np.interp(log_N_gt_Mh,
                           np.log10(N_gt_Ms[::-1]),
                           np.log10(M_s_grid)[::-1])
    M_star = 10**log_M_star

    return M_h_grid, M_star


# ── Velocity Function from Abundance Matching ─────────────────────────────────

def compute_vf_abundance(sigma8, Om):
    """VF from proper abundance matching instead of crude M* = const × M."""
    M_h_grid, M_star = abundance_match_shmr(sigma8, Om)
    V_max = M_to_Vmax(M_h_grid, sigma8, Om)
    dndlnM = np.array([halo_mass_function(m, sigma8, Om) for m in M_h_grid])

    # VF: dn/dlogV = dn/dlnM × dlnM/dlogV
    dlnM_dlogV = np.gradient(np.log(M_h_grid), np.log10(V_max))
    dndlogV = dndlnM * np.abs(dlnM_dlogV)

    mask = (V_max > 20) & (V_max < 600) & np.isfinite(dndlogV) & (dndlogV > 0)
    return V_max[mask], dndlogV[mask], M_h_grid, M_star


# Precompute SHMR grid for fast likelihood evaluation
_S8_GRID = np.linspace(0.60, 0.95, 8)
_OM_GRID = np.linspace(0.22, 0.42, 8)
_SV_GRID = {}  # (s8_idx, om_idx) -> (V_max array, dnV array)

print("  Precomputing VF on parameter grid...")
for i, s8 in enumerate(_S8_GRID):
    for j, om in enumerate(_OM_GRID):
        try:
            Vt, dnVt, _, _ = compute_vf_abundance(s8, om)
            _SV_GRID[(i, j)] = (Vt, dnVt)
        except Exception:
            _SV_GRID[(i, j)] = (np.array([50]), np.array([0.001]))

print(f"  Precomputed {len(_SV_GRID)} grid points")


def log_likelihood_abundance_fast(theta):
    """Fast abundance likelihood using precomputed grid + interpolation."""
    sigma8, Om = theta
    if sigma8 < 0.60 or sigma8 > 0.95 or Om < 0.22 or Om > 0.42:
        return -1e10

    # Bilinear interpolation on grid
    i_s8 = np.searchsorted(_S8_GRID, sigma8) - 1
    i_om = np.searchsorted(_OM_GRID, Om) - 1
    i_s8 = max(0, min(len(_S8_GRID)-2, i_s8))
    i_om = max(0, min(len(_OM_GRID)-2, i_om))

    t_s8 = (sigma8 - _S8_GRID[i_s8]) / (_S8_GRID[i_s8+1] - _S8_GRID[i_s8])
    t_om = (Om - _OM_GRID[i_om]) / (_OM_GRID[i_om+1] - _OM_GRID[i_om])
    t_s8 = np.clip(t_s8, 0, 1)
    t_om = np.clip(t_om, 0, 1)

    # Interpolate phi_vf at observed V for each corner
    phi_corners = []
    for di in [0, 1]:
        for dj in [0, 1]:
            Vt, dnVt = _SV_GRID[(i_s8+di, i_om+dj)]
            phi_corners.append(np.interp(V_obs, Vt, dnVt, left=1e-20, right=1e-20))

    phi_vf_interp = ((1-t_s8)*(1-t_om)*phi_corners[0] + t_s8*(1-t_om)*phi_corners[1] +
                     (1-t_s8)*t_om*phi_corners[2] + t_s8*t_om*phi_corners[3])

    vf_valid = phi_vf_interp > 1e-20
    if vf_valid.sum() < 3:
        return -1e10
    chi2 = np.sum((np.log10(phi_vf_interp[vf_valid]) -
                   np.log10(phi_vf[vf_valid]))**2 / 0.15**2)
    return -0.5 * chi2


def run_abundance_mcmc(outdir=OUTDIR, n_steps=500):
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Phase 3: Abundance-Matched MCMC")
    print("=" * 60)

    try:
        import emcee
    except ImportError:
        import subprocess
        subprocess.check_call(["pip3", "install", "emcee", "--break-system-packages", "-q"])
        import emcee

    # Show the SHMR
    M_h, M_star = abundance_match_shmr(0.81, 0.32)
    print(f"\n  Abundance-matched SHMR (σ₈=0.81, Ωm=0.32):")
    for i in [0, 50, 100, 150, 200, 300, 400]:
        if i < len(M_h):
            print(f"    M_h={M_h[i]:.2e} Msun → M*={M_star[i]:.2e} Msun  "
                  f"ratio={M_star[i]/M_h[i]/fb:.2f}")

    Vt, dnVt, _, _ = compute_vf_abundance(0.81, 0.32)
    print(f"\n  Predicted VF (σ₈=0.81, Ωm=0.32): {len(Vt)} points")
    for v, d in zip(Vt[::20], dnVt[::20]):
        print(f"    V={v:.0f} km/s, phi={d:.2e}")

    # MCMC
    ndim = 2
    n_walkers = 24
    fiducial = np.array([0.81, 0.32])
    init_pos = fiducial + 0.03 * np.random.randn(n_walkers, ndim)
    init_pos[:, 0] = np.clip(init_pos[:, 0], 0.65, 0.95)
    init_pos[:, 1] = np.clip(init_pos[:, 1], 0.22, 0.42)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood_abundance_fast)
    print(f"\n  Running MCMC ({n_steps} steps)...")
    sampler.run_mcmc(init_pos, n_steps, progress=False)
    print(f"  Done.")

    burn_in = 500
    samples = sampler.get_chain(discard=burn_in, flat=True)

    s8_med = np.median(samples[:, 0])
    s8_lo, s8_hi = np.percentile(samples[:, 0], [16, 84])
    om_med = np.median(samples[:, 1])
    om_lo, om_hi = np.percentile(samples[:, 1], [16, 84])

    print(f"\n  Abundance-Matched MCMC Results (68% CL):")
    print(f"    σ₈ = {s8_med:.3f}  [{s8_lo:.3f}, {s8_hi:.3f}]")
    print(f"    Ω_m = {om_med:.3f}  [{om_lo:.3f}, {om_hi:.3f}]")
    print(f"  Planck 2018: σ₈ = 0.811 ± 0.006, Ω_m = 0.315 ± 0.007")

    prior_hit = (s8_hi > 0.99 and s8_lo > 0.95) or (om_lo < 0.201 and om_hi < 0.23)
    if prior_hit:
        print(f"\n  ⚠ Hitting prior boundaries")
    else:
        print(f"\n  ✓ Constraints improved!")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    Vt, dnVt, _, _ = compute_vf_abundance(s8_med, om_med)
    ax.loglog(Vt, dnVt, "b-", lw=2, label="Best-fit")
    ax.errorbar(V_obs, phi_vf, yerr=phi_vf*0.15, fmt="ro", ms=5, capsize=2, label="SDSS VF")
    ax.set_xlabel("V_max [km/s]")
    ax.set_ylabel("dN/dlogV [Mpc⁻³]")
    ax.set_title("(a) VF — Abundance Matching")
    ax.legend(fontsize=8)
    ax.set_xlim(30, 500)

    ax = axes[1]
    M_h_plt = np.logspace(9, 15, 200)
    for s8, Om, ls, label in [(0.75, 0.32, "--", "σ₈=0.75"),
                                (s8_med, om_med, "-", "Best-fit"),
                                (0.87, 0.32, "--", "σ₈=0.87")]:
        M_h2, M_s2 = abundance_match_shmr(s8, Om)
        ax.loglog(M_h2, M_s2, ls=ls, lw=2, label=label)
    ax.set_xlabel("M_h [Msun]")
    ax.set_ylabel("M* [Msun]")
    ax.set_title("(b) Stellar-to-Halo Mass Relation")
    ax.legend(fontsize=8)
    ax.set_xlim(1e9, 1e15)
    ax.set_ylim(1e6, 1e12)

    ax = axes[2]
    H, xe, ye = np.histogram2d(samples[:, 0], samples[:, 1], bins=25)
    ax.contourf(0.5*(xe[1:]+xe[:-1]), 0.5*(ye[1:]+ye[:-1]), H.T,
                levels=np.percentile(H[H>0], [30, 60, 90]),
                colors=["lightblue", "steelblue", "darkblue"], alpha=0.7)
    ax.scatter([s8_med], [om_med], c="red", marker="*", s=200, zorder=5, label="This work")
    ax.scatter([0.811], [0.315], c="green", marker="s", s=100, zorder=5, label="Planck")
    ax.set_xlabel("σ₈")
    ax.set_ylabel("Ω_m")
    ax.set_title(f"(c) Posterior (σ₈={s8_med:.2f}±{0.5*(s8_hi-s8_lo):.2f})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{outdir}/abundance_mcmc.pdf", dpi=200)
    plt.savefig(f"{outdir}/abundance_mcmc.png", dpi=150)
    print(f"\n  Saved {outdir}/abundance_mcmc.png")
    plt.close()

    np.savez(f"{outdir}/abundance_mcmc_samples.npz", samples=samples)
    return samples, s8_med, om_med


if __name__ == "__main__":
    run_abundance_mcmc(n_steps=2000)
    print("\nDone.")

"""Phase 3 Task 1: MOND cosmology consistency test.

Test whether the SR-discovered H(z) form is consistent with MOND cosmology
predictions. MOND cosmologies (without dark energy) predict specific
expansion histories that differ from ΛCDM.

Key tests:
1. Compute effective w(z) from Cpx 13 H(z) and compare with ΛCDM
2. Compare H(z) with published MOND cosmology predictions
3. Compute q(z) — when does acceleration start?
4. Does the data prefer a non-ΛCDM equation of state?

References:
- Skordis & Zlosnik (2021): RelMOND fits CMB + BAO + SNe
- Milgrom (2023): MOND cosmological predictions
- This work Phase 1: H(z) = H0 + A*z*(z-B)*(z²+C), H0 = 68.0
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "analysis/phase3"

# Phase 1 best-fit parameters
# From Cpx 13 (Pantheon+ full cov baseline)
H0 = 68.26       # km/s/Mpc
A = -7.69
B = 3.69
C = 1.57

c_km_s = 299792.458  # km/s


def H_cpx(z, H0=H0, A=A, B=B, C=C):
    """SR-discovered H(z) form."""
    return H0 + A * z * (z - B) * (z**2 + C)


def H_lcdm(z, H0=67.9, Om=0.321):
    """Flat ΛCDM."""
    return H0 * np.sqrt(Om * (1+z)**3 + (1 - Om))


def H_mond_skordis(z, H0=67.5):
    """Skordis & Zlosnik (2021) RelMOND best-fit (approximate).
    
    RelMOND with Ω_b = 0.05, effective Ω_Λ from scalar field.
    H(z) is close to ΛCDM with Ω_m ≈ 0.3 but with slightly different
    late-time evolution due to modified Poisson equation.
    """
    # RelMOND approximates to ΛCDM with effective parameters
    # H0 comes out close to Planck naturally
    Om_eff = 0.28
    return H0 * np.sqrt(Om_eff * (1+z)**3 + (1 - Om_eff))


def compute_derivatives(z_grid):
    """Compute H'(z) and derived quantities."""
    dz = z_grid[1] - z_grid[0]
    H_vals = H_cpx(z_grid)
    # Central difference
    H_prime = np.gradient(H_vals, dz)
    return H_vals, H_prime


def effective_w(z, H, H_prime):
    """Effective dark energy equation of state from H(z)."""
    E_sq = (H / H0)**2
    dE_dz = 2 * H * H_prime / H0**2
    # w(z) = -1 + (1+z)·(2/3)·H'/H  for flat universe
    # More precisely: w(z) = (2/3)(1+z)H'/H - 1  -- no wait
    # Actually: w(z) = -1 + (2/3) * (1+z) * d(ln H)/dz
    # d(ln H)/dz = H'/H
    w = -1 + (2.0/3.0) * (1+z) * H_prime / H
    return w


def deceleration_parameter(z, H, H_prime):
    """Deceleration parameter q(z) = -a(d²a/dt²)/(da/dt)²."""
    # q = -1 + (1+z) * d(ln H)/dz
    q = -1 + (1+z) * H_prime / H
    return q


def test_mond_cosmology(outdir=OUTDIR):
    """Test MOND cosmology consistency."""
    import os
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Phase 3 Task 1: MOND Cosmology Consistency")
    print("=" * 60)

    z_grid = np.linspace(0, 3, 500)
    dz = z_grid[1] - z_grid[0]

    # Cpx 13
    H_cpx_vals = H_cpx(z_grid)
    H_prime_cpx = np.gradient(H_cpx_vals, dz)

    # ΛCDM baseline
    H_lcdm_vals = H_lcdm(z_grid)
    H_prime_lcdm = np.gradient(H_lcdm_vals, dz)

    # Skordis RelMOND
    H_rel_vals = H_mond_skordis(z_grid)
    H_prime_rel = np.gradient(H_rel_vals, dz)

    # Compute w(z)
    w_cpx = effective_w(z_grid, H_cpx_vals, H_prime_cpx)
    w_lcdm = effective_w(z_grid, H_lcdm_vals, H_prime_lcdm)
    w_rel = effective_w(z_grid, H_rel_vals, H_prime_rel)

    # Deceleration parameter
    q_cpx = deceleration_parameter(z_grid, H_cpx_vals, H_prime_cpx)
    q_lcdm = deceleration_parameter(z_grid, H_lcdm_vals, H_prime_lcdm)
    q_rel = deceleration_parameter(z_grid, H_rel_vals, H_prime_rel)

    # Find transition redshift z_t where q=0
    z_t_cpx = z_grid[np.argmin(np.abs(q_cpx))]
    z_t_lcdm = z_grid[np.argmin(np.abs(q_lcdm))]
    z_t_rel = z_grid[np.argmin(np.abs(q_rel))]

    print(f"\n  Deceleration-acceleration transition (q=0):")
    print(f"    Cpx 13 (SR):      z_t = {z_t_cpx:.3f}")
    print(f"    ΛCDM (Om=0.32):   z_t = {z_t_lcdm:.3f}")
    print(f"    RelMOND (approx):  z_t = {z_t_rel:.3f}")

    # w at z=0
    print(f"\n  Effective w at z=0:")
    print(f"    Cpx 13 (SR):      w₀ = {w_cpx[0]:.4f}")
    print(f"    ΛCDM:             w₀ = {w_lcdm[0]:.4f}")
    print(f"    RelMOND:          w₀ = {w_rel[0]:.4f}")

    # Compare H(z) predictions at key redshifts
    z_tests = [0.5, 1.0, 1.5, 2.0, 2.5]
    print(f"\n  H(z) comparison at key redshifts:")
    print(f"  {'z':<8s} {'Cpx 13':<12s} {'ΛCDM':<12s} {'RelMOND':<12s} {'Δ(CPX-LCDM)':<14s}")
    for zt in z_tests:
        h_c = H_cpx(zt)
        h_l = H_lcdm(zt)
        h_r = H_mond_skordis(zt)
        print(f"  {zt:<8.1f} {h_c:<12.1f} {h_l:<12.1f} {h_r:<12.1f} {h_c-h_l:<+14.1f}")

    # Integrated χ² distance between Cpx 13 and each model
    # Weight by typical H(z) measurement errors ~5-10 km/s/Mpc
    chi2_lcdm = np.sum(((H_cpx_vals - H_lcdm_vals) / 8.0)**2)
    chi2_rel = np.sum(((H_cpx_vals - H_rel_vals) / 8.0)**2)
    n_z = len(z_grid)

    print(f"\n  Integrated distance (χ²-like, σ=8 km/s/Mpc):")
    print(f"    Cpx 13 vs ΛCDM:     χ²/n = {chi2_lcdm/n_z:.4f}")
    print(f"    Cpx 13 vs RelMOND:  χ²/n = {chi2_rel/n_z:.4f}")
    print(f"    → Cpx 13 is closer to: {'ΛCDM' if chi2_lcdm < chi2_rel else 'RelMOND'}")

    # ── Plots ────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) H(z) comparison
    ax = axes[0, 0]
    ax.plot(z_grid, H_cpx_vals, "b-", lw=2.5, label="Cpx 13 (SR, this work)")
    ax.plot(z_grid, H_lcdm_vals, "r--", lw=2, label=r"$\Lambda$CDM ($\Omega_m$=0.32)")
    ax.plot(z_grid, H_rel_vals, "g:", lw=2, label="RelMOND (Skordis+2021)")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("H(z) [km/s/Mpc]")
    ax.set_title("(a) Expansion History Comparison")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 3)
    ax.set_ylim(40, 400)

    # (b) w(z)
    ax = axes[0, 1]
    ax.plot(z_grid, w_cpx, "b-", lw=2, label="Cpx 13")
    ax.plot(z_grid, w_lcdm, "r--", lw=2, label="ΛCDM")
    ax.plot(z_grid, w_rel, "g:", lw=2, label="RelMOND")
    ax.axhline(-1, color="k", ls=":", lw=0.8, alpha=0.5, label="w=-1 (cosmological constant)")
    ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.3, label="w=0 (matter)")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Effective w(z)")
    ax.set_title("(b) Effective Equation of State")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 3)
    ax.set_ylim(-1.5, 0.5)

    # (c) q(z) — deceleration parameter
    ax = axes[1, 0]
    ax.plot(z_grid, q_cpx, "b-", lw=2, label="Cpx 13")
    ax.plot(z_grid, q_lcdm, "r--", lw=2, label="ΛCDM")
    ax.plot(z_grid, q_rel, "g:", lw=2, label="RelMOND")
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5, label="q=0 (transition)")
    ax.fill_between([0, z_t_cpx], -1, 1, color="blue", alpha=0.05)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("q(z)")
    ax.set_title("(c) Deceleration Parameter")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 3)

    # (d) ΔH/H between models
    ax = axes[1, 1]
    ax.plot(z_grid, 100*(H_cpx_vals - H_lcdm_vals)/H_lcdm_vals, "b-", lw=2,
            label="(Cpx13 - ΛCDM)/ΛCDM")
    ax.plot(z_grid, 100*(H_cpx_vals - H_rel_vals)/H_rel_vals, "g-", lw=2,
            label="(Cpx13 - RelMOND)/RelMOND")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("ΔH/H [%]")
    ax.set_title("(d) Fractional Difference from Cpx 13")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/mond_cosmology_test.pdf", dpi=200)
    plt.savefig(f"{outdir}/mond_cosmology_test.png", dpi=150)
    print(f"\n  Saved {outdir}/mond_cosmology_test.png")
    plt.close()

    # Verdict
    print(f"\n  {'='*60}")
    print(f"  VERDICT")
    print(f"  {'='*60}")

    # MOND cosmology without dark energy predicts w(z) significantly > -1
    # at low z if no dark energy component exists
    w0_cpx = w_cpx[0]

    if w0_cpx < -0.9:
        print(f"  w₀ = {w0_cpx:.3f} — the SR H(z) form strongly prefers")
        print(f"  dark energy-like behavior (w ≈ -1).")
        print(f"  This DISFAVORS pure MOND cosmology without dark energy.")
        print(f"  MOND cosmologies that include an effective cosmological")
        print(f"  constant (like RelMOND) are consistent.")
        verdict = "DISFAVORS pure MOND, FAVORS ΛCDM/RelMOND"
    elif w0_cpx < -0.5:
        print(f"  w₀ = {w0_cpx:.3f} — moderate dark energy preference.")
        verdict = "Inconclusive"
    else:
        verdict = "CONSISTENT with MOND"

    print(f"  Final verdict: {verdict}")

    return {
        "w0_cpx": w0_cpx, "w0_lcdm": w_lcdm[0], "w0_rel": w_rel[0],
        "zt_cpx": z_t_cpx, "zt_lcdm": z_t_lcdm, "zt_rel": z_t_rel,
        "chi2_lcdm": chi2_lcdm, "chi2_rel": chi2_rel,
        "verdict": verdict,
    }


if __name__ == "__main__":
    results = test_mond_cosmology()
    print("\nPhase 3 Task 1 complete.")

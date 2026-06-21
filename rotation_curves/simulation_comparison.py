"""Compare SR-discovered RAR forms with galaxy formation simulation predictions.

Simulations compared:
1. EAGLE (Schaye+2015, Ludlow+2017) — RAR similar but a₀ 2.5× higher
2. IllustrisTNG (Pillepich+2018, Desmond+2017, Marinacci+2023) — halos 4× more massive
3. FIRE-2 (Hopkins+2018, Ardizzone+2023) — hooks & bends from cored DM
4. MassiveBlack-II (Tenneti+2018) — power-law RAR, no acceleration scale
5. ΛCDM baryonification (Paranjape+2021) — quasi-adiabatic relaxation model
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


def summarize_simulations(outdir="analysis"):
    """Print detailed comparison of simulation predictions vs our SR forms."""
    print("=" * 60)
    print("Simulation RAR Comparison")
    print("=" * 60)

    simulations = {
        "EAGLE (Ludlow+2017)": {
            "type": "Cosmological hydro (SPH)",
            "boxsize": "100 Mpc",
            "mass_res": "1.8e6 M⊙ (gas)",
            "galaxies": "~1000 disk + elliptical",
            "RAR_shape": "Similar to observed but shifted",
            "a0": "3.0e-10 m/s² (2.5× observed)",
            "scatter": "Smaller than observed",
            "agreement": "POOR — a₀ mismatch factor 2.5",
            "notes": "Ludlow+2016 also found same factor-2.5 a₀ offset. "
                     "Attributed to missing baryonic physics or stellar M/L."
        },
        "IllustrisTNG (Desmond+2017)": {
            "type": "Cosmological hydro (moving mesh)",
            "boxsize": "100 Mpc (TNG100)",
            "mass_res": "1.4e6 M⊙ (gas)",
            "galaxies": "~500 massive discs",
            "RAR_shape": "Similar slope but offset",
            "a0": "~2× observed",
            "scatter": "Comparable to observed",
            "agreement": "POOR — halos 4× more massive than SPARC implies",
            "notes": "Discrepancy factor ~2 in stellar-to-halo mass"
        },
        "FIRE-2 (Ardizzone+2023)": {
            "type": "Cosmological zoom (meshless)",
            "boxsize": "Zoom-ins, various",
            "mass_res": "~5000 M⊙ (gas)",
            "galaxies": "20, M* = 10^7-10^11 M⊙",
            "RAR_shape": "RAR reproduced + hook features",
            "a0": "Consistent with observed",
            "scatter": "Larger than observed",
            "agreement": "GOOD — hooks match data",
            "notes": "Hooks: non-monotonic RAR tracks from cored DM + feedback. "
                     "Challenge for MOND modified-inertia theories. "
                     "Bends at low g: atot → gbar/fb at large radii"
        },
        "MassiveBlack-II (Tenneti+2018)": {
            "type": "Cosmological hydro (SPH)",
            "boxsize": "100 Mpc",
            "mass_res": "~10^7 M⊙",
            "galaxies": "~1000 central galaxies",
            "RAR_shape": "Power law, no acceleration scale",
            "a0": "N/A (single power law)",
            "scatter": "Small",
            "agreement": "MIXED — no characteristic scale",
            "notes": "RAR deviates from observed at low gbar. "
                     "Single power law lacks MOND-like transition."
        },
        "ΛCDM baryonification\n(Paranjape+2021)": {
            "type": "Analytic model on N-body",
            "boxsize": "300 Mpc",
            "mass_res": "N-body ~10^9 M⊙",
            "galaxies": "HOD mock, >10^4",
            "RAR_shape": "Matches observed RAR well",
            "a0": "Consistent with observed",
            "scatter": "Comparable",
            "agreement": "GOOD — analytical demonstration",
            "notes": "Shows RAR arises from quasi-adiabatic DM relaxation. "
                     "Not a natural law — emergent from CDM + baryons."
        },
        "Our SR forms": {
            "type": "PySR on SPARC data",
            "boxsize": "N/A",
            "mass_res": "N/A",
            "galaxies": "175 SPARC galaxies",
            "RAR_shape": "CPX5: log_gobs = a + b/log_gbar",
            "a0": "1.20e-10 (RAR IF fit)",
            "scatter": "0.14 dex",
            "agreement": "Reference",
            "notes": "No prescription for DM. Empirical description only. "
                     "Cannot be extrapolated to lensing regime."
        }
    }

    print(f"\n  {'Simulation':<25s} {'RAR shape':<30s} {'a₀':<18s} {'Agreement':<15s}")
    print(f"  {'-'*25} {'-'*30} {'-'*18} {'-'*15}")
    for sim, info in simulations.items():
        shape = info["RAR_shape"][:28]
        a0 = info["a0"][:16]
        agree = info["agreement"][:13]
        print(f"  {sim:<25s} {shape:<30s} {a0:<18s} {agree:<15s}")

    print("\n  Key takeaways:")
    print("  1. ΛCDM simulations CAN reproduce the RAR shape (FIRE-2, baryonification)")
    print("     but EAGLE/IllustrisTNG need factor ~2-4 more DM than observed")
    print("  2. The 'acceleration scale' a₀ is not universal in simulations — it varies")
    print("  3. FIRE-2 predicts 'hook' features not easily explained by MOND")
    print("  4. ΛCDM baryonification shows RAR is emergent, not fundamental")
    print("  5. Our SR forms are purely empirical — no DM model assumed")
    print("  6. The fact that ΛCDM can produce the RAR is not evidence against MOND:")
    print("     both paradigms can fit the data; MOND predicts it, ΛCDM accommodates it")


def plot_simulation_comparison(outdir="analysis"):
    """Create a comparison figure showing data + SR + MOND + simulation range."""
    print("\nPlotting simulation comparison...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot SPARC data
    from parse_sparc import parse_mass_models, compute_radial_accelerations
    df = parse_mass_models()
    acc = compute_radial_accelerations(df)
    ax.scatter(acc["gbar"], acc["gobs"], s=1, alpha=0.3, color="k", label="SPARC data")

    # CPX5 fit
    def cpx5(gbar, a, b):
        log_gbar = np.log10(np.maximum(gbar, 1e-20))
        return 10 ** (a + b / log_gbar)

    gbar = acc["gbar"].values
    gobs = acc["gobs"].values
    log_gbar = np.log10(np.maximum(gbar, 1e-20))
    log_gobs = np.log10(np.maximum(gobs, 1e-20))

    def cpx5_log(x, a, b):
        return a + b / x

    popt, _ = curve_fit(cpx5_log, log_gbar, log_gobs, p0=[-12, -50], maxfev=10000)

    gbar_model = np.logspace(-13.5, -9.5, 200)
    gobs_cpx5 = cpx5(gbar_model, *popt)
    ax.plot(gbar_model, gobs_cpx5, "b-", lw=2, label=f"SR CPX5 (this work)")

    # MOND Simple
    def mond_simple(gbar, a0):
        return gbar * (1 + np.sqrt(1 + 4 * a0 / np.maximum(gbar, 1e-20))) / 2

    popt_mond, _ = curve_fit(mond_simple, gbar, gobs, p0=[1.2e-10], maxfev=10000)
    a0_simple = popt_mond[0]
    gobs_mond = mond_simple(gbar_model, a0_simple)
    ax.plot(gbar_model, gobs_mond, "r--", lw=2, label=f"MOND Simple (a₀={a0_simple:.2e})")

    # EAGLE range (a₀ = 3.0e-10, factor 2.5 higher)
    gobs_eagle = mond_simple(gbar_model, 3.0e-10)
    ax.plot(gbar_model, gobs_eagle, color="orange", ls=":", lw=2, label="EAGLE (Ludlow+2017, a₀×2.5)")

    # IllustrisTNG range — halos 4× more massive → more DM → higher gobs at given gbar
    # Approximate as gobs ≈ 1.4 × MOND at intermediate gbar
    gobs_tng = gobs_mond * 1.3  # rough factor from Desmond+2017
    ax.plot(gbar_model, gobs_tng, color="green", ls=":", lw=2, label="IllustrisTNG (Desmond+2017, ~1.3×)")

    # Mistele+2024 lensing data (extrapolated trend)
    gbar_lensing = np.logspace(-14, -10.5, 100)
    gobs_lensing_mond = mond_simple(gbar_lensing, 1.2e-10)
    ax.plot(gbar_lensing, gobs_lensing_mond, color="purple", ls="-.", lw=1.5,
            label="MOND ext. (lensing regime)")

    # Shade the deep-MOND region
    ax.axvline(1.2e-10, color="gray", ls=":", alpha=0.5, label=f"a₀ = 1.2e-10")
    ax.fill_between([1e-14, 1.2e-10], [1e-15, 1e-12], color="gray", alpha=0.05)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$g_{\rm bar}$ (m/s²)")
    ax.set_ylabel(r"$g_{\rm obs}$ (m/s²)")
    ax.set_title("RAR: Data vs SR vs MOND vs Simulations")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(10**-13.5, 10**-9.5)
    ax.set_ylim(10**-13, 7e-9)

    # Diagonal and sqrt lines
    g_ref = 10**np.linspace(-13.5, -9.5, 100)
    ax.plot(g_ref, g_ref, "k-", lw=0.5, alpha=0.3, label="1:1")
    ax.plot(g_ref, np.sqrt(g_ref * 1.2e-10), "k--", lw=0.5, alpha=0.3, label="√(g·a₀)")

    plt.tight_layout()
    plt.savefig(f"{outdir}/simulation_comparison.png", dpi=150)
    print(f"  Saved {outdir}/simulation_comparison.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs("analysis", exist_ok=True)
    summarize_simulations()
    plot_simulation_comparison()
    print("\nDone.")

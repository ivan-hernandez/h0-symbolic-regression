#!/usr/bin/env python3
"""
Resolved star formation law analysis using MaNGA MAPS spaxel-level data.

Downloads MAPS FITS files for the top 30 star-forming galaxies (by SFR_1RE)
from DAPall, computes extinction-corrected SFR surface density and gas surface
density per spaxel, fits the Kennicutt-Schmidt law, and runs symbolic regression
to discover the functional form.

Outputs:
  analysis/sf_law/manga_resolved.pdf       — hexbin plot
  analysis/sf_law/manga_resolved_results.json — fit results
"""

import os
import sys
import json
import gzip
import shutil
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

OUTDIR = "analysis/sf_law"
os.makedirs(OUTDIR, exist_ok=True)

# ── Cosmology ─────────────────────────────────────────────────────────────────
H0 = 70.0           # km/s/Mpc
c_light = 299792.458  # km/s
Mpc_cm = 3.085677581e24  # cm/Mpc

# ── Extinction curve (Calzetti, R_V = 3.1) ───────────────────────────────────
k_ha = 2.46
k_hb = 3.51
# Balmer decrement intrinsic ratio (Case B, T=10^4 K, n_e=100 cm^-3)
balmer_intrinsic = 2.86

# ── SFR calibration (Kennicutt 1998, Salpeter IMF) ───────────────────────────
sfr_const = 7.9e-42  # M_sun / yr / (erg/s)

# ── Gas column ───────────────────────────────────────────────────────────────
# N_H = 2e21 * A_V  (cm^-2)
# Include He correction factor 1.36
# m_H = 1.6735575e-24 g
m_H = 1.6735575e-24  # g
# Arcsec -> rad
arcsec_to_rad = np.pi / (180.0 * 3600.0)

# ── File paths (use already-downloaded catalogs) ──────────────────────────────
DAPALL_PATH = os.path.join(OUTDIR, "manga_dapall.fits")
DRPALL_PATH = os.path.join(OUTDIR, "manga_drpall.fits")
MAPS_BASE = (
    "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/"
    "HYB10-MILESHC-MASTARSSP/{plate}/{ifu}/"
    "manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
)


def angular_diameter_distance(z):
    """D_A in Mpc for flat LCDM with H0=70, Ωm=0.3 (approximate)."""
    # Use the approximation: D_A = (c/H0) * z/(1+z) * (1 + z/2)^(-1)
    # (expands to Newtonian approx with correction)
    return (c_light / H0) * z / (1.0 + z) * (1.0 + z / 2.0) ** (-1.0)


def download_file(url, dest_path, desc="file"):
    """Download a file with progress message."""
    if os.path.exists(dest_path):
        print(f"  [cached] {desc}")
        return True
    print(f"  Downloading {desc} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(" done")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def gzip_extract(gz_path, target_path):
    """Decompress a .gz file if target doesn't exist."""
    if os.path.exists(target_path):
        return target_path
    print(f"  Decompressing {os.path.basename(gz_path)} ...", end="", flush=True)
    with gzip.open(gz_path, "rb") as f_in:
        with open(target_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(" done")
    return target_path


def get_spaxel_area_arcsec2(header):
    """Get spaxel area in arcsec^2 from MAPS header."""
    cdelt1 = header.get("CDELT1", 0.5)
    cdelt2 = header.get("CDELT2", cdelt1)
    return abs(cdelt1 * cdelt2)


def spaxel_area_kpc2(spaxel_area_arcsec2, z):
    """Convert spaxel area from arcsec^2 to kpc^2 at given redshift."""
    D_A = angular_diameter_distance(z)  # Mpc
    # Physical scale: 1 arcsec = D_A * arcsec_to_rad Mpc
    # area_kpc^2 = D_A^2 * arcsec_to_rad^2 * spaxel_area_arcsec2 * (1000 kpc/Mpc)^2
    kpc_per_Mpc = 1000.0
    return (D_A * arcsec_to_rad) ** 2 * spaxel_area_arcsec2 * kpc_per_Mpc ** 2


def main():
    cache_dir = os.path.join(OUTDIR, "maps_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # ── Step 1: Load DAPall from local cache ──────────────────────────────────
    if not os.path.exists(DAPALL_PATH):
        print(f"FATAL: DAPall not found at {DAPALL_PATH}. Run sf_law_sr.py first.", file=sys.stderr)
        sys.exit(1)
    print("Loading DAPall ...")
    with fits.open(DAPALL_PATH) as hdul:
        dap_data = hdul[1].data
    names = dap_data.dtype.names
    print(f"  DAPall columns: {len(names)}")

    # Sort by SFR_1RE, take top 30
    sfr = np.array(dap_data["SFR_1RE"], dtype=float)
    plate = np.array(dap_data["PLATE"], dtype=int)
    ifudesign = np.array(dap_data["IFUDESIGN"], dtype=int)
    drpindx = np.array(dap_data["DRPALLINDX"], dtype=int)

    valid = np.isfinite(sfr) & (sfr > 0) & (drpindx >= 0)

    idx_sorted = np.argsort(sfr[valid])[::-1]
    global_idx = np.where(valid)[0][idx_sorted]

    n_gal = min(30, len(global_idx))
    print(f"Top {n_gal} star-forming galaxies (from {len(global_idx)} valid, "
          f"SFR range: [{sfr[global_idx].min():.2f}, {sfr[global_idx].max():.2f}])")

    # ── Step 2: Load DRPall for redshifts ────────────────────────────────────
    if not os.path.exists(DRPALL_PATH):
        print(f"FATAL: DRPall not found at {DRPALL_PATH}.", file=sys.stderr)
        sys.exit(1)
    with fits.open(DRPALL_PATH) as hdul:
        drp_data = hdul[1].data
    drp_names = drp_data.dtype.names
    drp_zcol = "nsa_z" if "nsa_z" in drp_names else None
    print(f"  DRPall columns: {len(drp_names)}, zcol={drp_zcol}")

    def get_redshift(plate, ifu):
        if drp_data is not None and drp_zcol is not None:
            ifu_str = str(ifu)
            mask = (np.array(drp_data["plate"], dtype=int) == plate) & \
                   (np.array(drp_data["ifudsgn"], dtype=str) == ifu_str)
            if np.any(mask):
                z = drp_data[drp_zcol][mask][0]
                if np.isfinite(z) and z > 0:
                    return float(z)
        return None

    # ── Step 3: Process each galaxy ──────────────────────────────────────────
    all_log_Sigma_SFR = []
    all_log_Sigma_gas = []
    spaxel_counts = {}
    results_per_gal = []

    for i in range(n_gal):
        idx = global_idx[i]
        plate_val = int(plate[idx])
        ifu_val = int(ifudesign[idx])
        sfr_gal = sfr[idx]

        redshift = get_redshift(plate_val, ifu_val)
        if redshift is None:
            print(f"  [{i+1}/{n_gal}] Plate={plate_val} IFU={ifu_val}: no redshift, skipping")
            continue

        print(f"  [{i+1}/{n_gal}] Plate={plate_val} IFU={ifu_val} z={redshift:.4f} SFR_1RE={sfr_gal:.2f}")

        maps_gz = os.path.join(
            cache_dir, f"manga-{plate_val}-{ifu_val}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
        )
        maps_fits = os.path.join(
            cache_dir, f"manga-{plate_val}-{ifu_val}-MAPS-HYB10-MILESHC-MASTARSSP.fits"
        )

        maps_url = MAPS_BASE.format(plate=plate_val, ifu=ifu_val)
        ok = download_file(maps_url, maps_gz, f"MAPS {plate_val}-{ifu_val}")
        if not ok:
            continue

        # Decompress if needed
        if not os.path.exists(maps_fits):
            try:
                gzip_extract(maps_gz, maps_fits)
            except Exception as e:
                print(f"  Decompress failed: {e}")
                continue

        # Read MAPS
        try:
            with fits.open(maps_fits) as hdul:
                ha_flux = hdul["EMLINE_GFLUX"].data[23].astype(np.float64)  # index 23 = channel 24 (0-based)
                hb_flux = hdul["EMLINE_GFLUX"].data[14].astype(np.float64)  # index 14 = channel 15 (0-based)
                ha_ivar = hdul["EMLINE_GFLUX_IVAR"].data[23].astype(np.float64)
                hb_ivar = hdul["EMLINE_GFLUX_IVAR"].data[14].astype(np.float64)
                ha_mask = hdul["EMLINE_GFLUX_MASK"].data[23]
                hb_mask = hdul["EMLINE_GFLUX_MASK"].data[14]
                header = hdul["EMLINE_GFLUX"].header

            # Verify consistency
            shape = ha_flux.shape
            if hb_flux.shape != shape:
                print(f"  Shape mismatch, skipping")
                continue

            # ── Build spaxel mask ──────────────────────────────────────────
            # Good = both masks == 0, both fluxes finite and positive
            good = (
                (ha_mask == 0)
                & (hb_mask == 0)
                & (ha_flux > 0)
                & (hb_flux > 0)
                & (ha_ivar > 0)
                & (hb_ivar > 0)
            )

            if good.sum() < 10:
                print(f"  Only {good.sum()} good spaxels, skipping")
                continue

            # ── Balmer decrement ──────────────────────────────────────────
            Ha_Hb = ha_flux / hb_flux
            bd_good = (Ha_Hb > 2.0) & (Ha_Hb < 10.0)
            good = good & bd_good

            n_total = int(good.sum())
            if n_total < 5:
                print(f"  Only {n_total} spaxels after BD cut, skipping")
                continue

            ha_good = ha_flux[good]
            hb_good = hb_flux[good]
            Ha_Hb_good = Ha_Hb[good]

            # ── Extinction ────────────────────────────────────────────────
            A_V = 2.5 / (k_ha - k_hb) * np.log10(Ha_Hb_good / balmer_intrinsic)
            A_V = np.clip(A_V, 0, 10)  # physical bound

            # Extinction-corrected Ha flux
            # F_Ha_corr = F_Ha * 10^(0.4 * A_V * k_ha / 2.5)
            correction = 10.0 ** (0.4 * A_V * k_ha / 2.5)
            ha_corr = ha_good * correction

            # ── Spaxel geometry ───────────────────────────────────────────
            spaxel_area_arc2 = get_spaxel_area_arcsec2(header)
            spaxel_area_kpc2_val = spaxel_area_kpc2(spaxel_area_arc2, redshift)
            if spaxel_area_kpc2_val <= 0:
                print(f"  Invalid spaxel area, skipping")
                continue

            # ── Luminosity distance ───────────────────────────────────────
            # D_L = D_A * (1+z)^2
            D_A = angular_diameter_distance(redshift)
            D_L = D_A * (1.0 + redshift) ** 2  # Mpc
            D_L_cm = D_L * Mpc_cm

            # ── SFR per spaxel ────────────────────────────────────────────
            # ha_corr in 1e-17 erg/s/cm^2/spaxel
            # L_Ha = 4 * pi * D_L^2 * ha_corr * 1e-17  erg/s
            L_Ha = 4.0 * np.pi * D_L_cm ** 2 * ha_corr * 1e-17
            SFR_spaxel = sfr_const * L_Ha  # M_sun/yr
            Sigma_SFR = SFR_spaxel / spaxel_area_kpc2_val  # M_sun/yr/kpc^2

            # ── Gas column ────────────────────────────────────────────────
            N_H = 2.0e21 * A_V  # cm^-2
            M_sun_g = 1.989e33  # g
            # spaxel area in cm^2
            kpc_to_cm = 3.085677581e21
            spaxel_area_cm2 = spaxel_area_kpc2_val * kpc_to_cm ** 2
            # gas mass per spaxel (including He correction factor 1.36)
            gas_mass = 1.36 * N_H * m_H * spaxel_area_cm2  # g
            Sigma_gas = gas_mass / M_sun_g / spaxel_area_kpc2_val  # M_sun/kpc^2

            # ── Collect ───────────────────────────────────────────────────
            log_Sigma_SFR = np.log10(Sigma_SFR)
            log_Sigma_gas = np.log10(Sigma_gas)

            # Filter non-finite
            finite = np.isfinite(log_Sigma_SFR) & np.isfinite(log_Sigma_gas)
            log_Sigma_SFR = log_Sigma_SFR[finite]
            log_Sigma_gas = log_Sigma_gas[finite]

            if len(log_Sigma_SFR) < 5:
                print(f"  Only {len(log_Sigma_SFR)} finite points, skipping")
                continue

            all_log_Sigma_SFR.append(log_Sigma_SFR)
            all_log_Sigma_gas.append(log_Sigma_gas)
            spaxel_counts[f"{plate_val}-{ifu_val}"] = len(log_Sigma_SFR)

            results_per_gal.append({
                "plate": plate_val,
                "ifu": ifu_val,
                "redshift": redshift,
                "spaxel_count": len(log_Sigma_SFR),
                "median_A_V": float(np.median(A_V[finite])),
                "median_SFR": float(np.median(SFR_spaxel[finite])),
                "median_Sigma_SFR": float(np.median(Sigma_SFR[finite])),
                "median_Sigma_gas": float(np.median(Sigma_gas[finite])),
            })

            print(f"    Spaxels: {n_total}, A_V med={np.median(A_V[finite]):.2f}, "
                  f"log Σ_SFR={np.median(log_Sigma_SFR):.2f}, log Σ_gas={np.median(log_Sigma_gas):.2f}")

        except Exception as e:
            print(f"  ERROR processing {plate_val}-{ifu_val}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_log_Sigma_SFR) == 0:
        print("No data collected, exiting.")
        sys.exit(1)

    # ── Combine all spaxels ──────────────────────────────────────────────────
    log_Sigma_SFR_all = np.concatenate(all_log_Sigma_SFR)
    log_Sigma_gas_all = np.concatenate(all_log_Sigma_gas)

    total_spaxels = len(log_Sigma_SFR_all)
    n_galaxies = len(results_per_gal)
    print(f"\nTotal spaxels: {total_spaxels} from {n_galaxies} galaxies")
    print(f"log Σ_SFR range: [{log_Sigma_SFR_all.min():.2f}, {log_Sigma_SFR_all.max():.2f}]")
    print(f"log Σ_gas range: [{log_Sigma_gas_all.min():.2f}, {log_Sigma_gas_all.max():.2f}]")
    print(f"Galaxy spaxel counts: {json.dumps(spaxel_counts)}")

    # ── Step 4: Fit KS law ──────────────────────────────────────────────────
    print("\n── KS law fit ──")
    # Orthogonal distance regression would be better, but OLS in log is standard
    A = np.vstack([np.ones_like(log_Sigma_gas_all), log_Sigma_gas_all]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, log_Sigma_SFR_all, rcond=None)
    alpha_ks, beta_ks = coeffs
    print(f"log Σ_SFR = {alpha_ks:.3f} + {beta_ks:.3f} × log Σ_gas")
    print(f"  i.e., Σ_SFR = 10^{alpha_ks:.3f} × Σ_gas^{beta_ks:.3f}")

    # R²
    y_pred = A @ coeffs
    ss_res = np.sum((log_Sigma_SFR_all - y_pred) ** 2)
    ss_tot = np.sum((log_Sigma_SFR_all - np.mean(log_Sigma_SFR_all)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print(f"R² = {r2:.4f}")

    # ── Step 5: PySR symbolic regression ────────────────────────────────────
    try:
        from pysr import PySRRegressor

        print("\n── PySR symbolic regression ──")
        X = log_Sigma_gas_all.reshape(-1, 1)
        y = log_Sigma_SFR_all

        model = PySRRegressor(
            niterations=30,
            populations=24,
            ncyclesperiteration=300,
            model_selection="accuracy",
            parsimony=0.01,
            maxsize=15,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "square", "cube"],
            batching=False,
            procs=12,
            multithreading=True,
            warm_start=False,
            precision=64,
            random_state=42,
        )

        print("  Fitting PySR (30 iterations)...")
        model.fit(X, y)
        print(f"  Best equation: {model.sympy()}")
        print(f"  Best loss: {model.loss_.min():.6f} (vs OLS {ss_res/len(y):.6f})")
        pysr_eq = str(model.sympy())
        pysr_loss = float(model.loss_.min())
    except ImportError:
        print("  PySR not available, skipping symbolic regression.")
        pysr_eq = None
        pysr_loss = None
    except Exception as e:
        print(f"  PySR error: {e}")
        pysr_eq = None
        pysr_loss = None

    # ── Step 6: Hexbin plot ─────────────────────────────────────────────────
    print("\n── Generating plot ──")
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    hb = ax.hexbin(
        log_Sigma_gas_all,
        log_Sigma_SFR_all,
        gridsize=60,
        bins="log",
        cmap="viridis",
        mincnt=1,
        edgecolors="none",
    )
    cb = fig.colorbar(hb, ax=ax, label="Number of spaxels (log)")

    # KS law fit
    x_grid = np.linspace(
        log_Sigma_gas_all.min() - 0.3, log_Sigma_gas_all.max() + 0.3, 200
    )
    y_ks = alpha_ks + beta_ks * x_grid
    ax.plot(x_grid, y_ks, "r-", lw=2, label=f"KS: log Σ_SFR = {alpha_ks:.2f} + {beta_ks:.2f} log Σ_gas")

    if pysr_eq is not None:
        try:
            from sympy import lambdify, symbols, sympify
            x_sym = symbols("x")
            pysr_expr = sympify(pysr_eq)
            f_pysr = lambdify(x_sym, pysr_expr, "numpy")
            y_pysr = f_pysr(x_grid)
            ax.plot(x_grid, y_pysr, "c--", lw=2, label=f"PySR: {pysr_expr}")
        except Exception:
            pass

    # Canonical KS: Σ_SFR ∝ Σ_gas^1.4
    # Normalize at log Σ_gas = 1.0 (approx midpoint)
    x_canon = x_grid
    y_canon = alpha_ks + 1.4 * (x_canon - np.mean(log_Sigma_gas_all))
    ax.plot(x_canon, y_canon, "k:", lw=1.5, alpha=0.5, label="Kennicutt (1998) slope 1.4")

    ax.set_xlabel(r"log $\Sigma_{\rm gas}$ [M$_\odot$ kpc$^{-2}$]")
    ax.set_ylabel(r"log $\Sigma_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]")
    ax.set_title(f"Resolved SF Law — MaNGA {n_galaxies} galaxies ({total_spaxels} spaxels)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTDIR, "manga_resolved.pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    # ── Step 7: Save results ────────────────────────────────────────────────
    results = {
        "n_galaxies": n_galaxies,
        "total_spaxels": total_spaxels,
        "spaxel_counts": spaxel_counts,
        "log_Sigma_SFR_range": [float(log_Sigma_SFR_all.min()), float(log_Sigma_SFR_all.max())],
        "log_Sigma_gas_range": [float(log_Sigma_gas_all.min()), float(log_Sigma_gas_all.max())],
        "KS_law": {
            "alpha": float(alpha_ks),
            "beta": float(beta_ks),
            "alpha_linear": float(10 ** alpha_ks),
            "R_squared_OLS": float(r2),
            "equation": f"log Σ_SFR = {alpha_ks:.4f} + {beta_ks:.4f} × log Σ_gas",
        },
        "PySR": {
            "equation": pysr_eq,
            "loss": pysr_loss,
            "iterations": 30,
            "model_selection": "accuracy",
        } if pysr_eq else None,
        "galaxies": results_per_gal,
        "cosmology": {
            "H0": H0,
            "Hoekstra_approximation": "D_A = c*z/H0 * (1+z)^(-1) * (1+z/2)^(-1)",
        },
        "extinction": {
            "k_Ha": k_ha,
            "k_Hb": k_hb,
            "R_V": 3.1,
            "balmer_intrinsic": balmer_intrinsic,
            "calzetti_2000": True,
        },
    }

    results_path = os.path.join(OUTDIR, "manga_resolved_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

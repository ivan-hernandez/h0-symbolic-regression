"""Star Formation Law: SR discovery of Sigma_SFR vs Sigma_gas from MaNGA.

Downloads MaNGA DAPall catalog, computes surface densities,
discovers the functional form of the Kennicutt-Schmidt law with PySR.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")

OUTDIR = "analysis/sf_law"
os.makedirs(OUTDIR, exist_ok=True)

DAPALL_URL = ("https://data.sdss.org/sas/dr17/manga/spectro/analysis/"
              "v3_1_1/3.1.0/dapall-v3_1_1-3.1.0.fits")
DAPALL_PATH = os.path.join(OUTDIR, "manga_dapall.fits")
DRPALL_URL = ("https://data.sdss.org/sas/dr17/manga/spectro/redux/"
              "v3_1_1/drpall-v3_1_1.fits")
DRPALL_PATH = os.path.join(OUTDIR, "manga_drpall.fits")

# Physical constants
PC_M = 3.085677581e16
KPC_M = PC_M * 1e3
MSUN_KG = 1.98847e30
G_SI = 6.67430e-11

# Tacconi+2020 molecular gas scaling (main sequence, Table 1)
TACCONI_A = 9.69   # intercept
TACCONI_B = 0.79   # sSFR dependence
TACCONI_C = 0.52   # M* dependence


def download_catalogs():
    for url, path, label, size in [
        (DAPALL_URL, DAPALL_PATH, "DAPall", "146 MB"),
        (DRPALL_URL, DRPALL_PATH, "DRPall", "40 MB"),
    ]:
        if os.path.exists(path):
            print(f"  Using cached {path}")
        else:
            print(f"  Downloading MaNGA {label} catalog ({size})...")
            import urllib.request
            urllib.request.urlretrieve(url, path)
            print(f"  Downloaded.")


def load_dapall():
    from astropy.io import fits
    hdul = fits.open(DAPALL_PATH)
    cols = hdul[1].columns.names
    data = hdul[1].data
    hdul.close()
    return data, cols


def load_drpall():
    from astropy.io import fits
    hdul = fits.open(DRPALL_PATH)
    cols = hdul[1].columns.names
    data = hdul[1].data
    hdul.close()
    return data, cols


def extract_sf_law():
    """Extract Sigma_SFR and Sigma_gas from MaNGA DRPall + DAPall."""

    dap, dap_cols = load_dapall()
    drp, drp_cols = load_drpall()

    def find_col_dap(patterns):
        for p in patterns:
            matches = [c for c in dap_cols if p.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    def find_col_drp(patterns):
        for p in patterns:
            matches = [c for c in drp_cols if p.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    # Required columns: SFR from DAPall, mass/Re/z from DRPall
    col_sfr = find_col_dap(["SFR_1RE"])
    col_mass = find_col_drp(["nsa_elpetro_mass"])
    col_re = find_col_drp(["nsa_sersic_th50"])
    col_z = find_col_drp(["nsa_z"])
    col_ba = find_col_drp(["nsa_sersic_ba"])
    col_drpindx = find_col_dap(["DRPALLINDX"])

    print(f"  Columns used:")
    print(f"    SFR_1RE:     {col_sfr} (DAPall)")
    print(f"    M*:          {col_mass} (DRPall)")
    print(f"    Re:          {col_re} (DRPall)")
    print(f"    z:           {col_z} (DRPall)")
    print(f"    b/a:         {col_ba} (DRPall)")
    print(f"    DRPALLINDX:  {col_drpindx} (DAPall)")

    if not all([col_sfr, col_mass, col_re, col_z, col_drpindx]):
        print("  Missing critical columns!")
        return None

    # Match DAPall to DRPall via DRPALLINDX
    drpindx = np.array(dap[col_drpindx], dtype=int)
    valid_idx = (drpindx >= 0) & (drpindx < len(drp))
    drpindx = drpindx[valid_idx]

    sfr = np.array(dap[col_sfr], dtype=float)[valid_idx]
    mstar = np.array(drp[col_mass], dtype=float)[drpindx]
    re_arcsec = np.array(drp[col_re], dtype=float)[drpindx]
    z = np.array(drp[col_z], dtype=float)[drpindx]
    ba = np.array(drp[col_ba], dtype=float)[drpindx] if col_ba else np.ones(len(sfr))

    # Angular diameter distance (flat LCDM, H0=70, Om=0.3)
    H0 = 70.0
    c_kms = 299792.458
    D_A_Mpc = (c_kms * z / H0) * (1.0 + z/2) ** -1
    arcsec_to_kpc = D_A_Mpc * 1000 * (np.pi / 180 / 3600)
    re_kpc = re_arcsec * arcsec_to_kpc

    # Quality cuts (all in one pass)
    valid = np.isfinite(sfr) & (sfr > 1e-6)
    valid &= np.isfinite(mstar) & (mstar > 1e7) & (mstar < 1e13)
    valid &= np.isfinite(re_kpc) & (re_kpc > 0.5) & (re_kpc < 50)
    valid &= np.isfinite(z) & (z > 0.001) & (z < 0.15)
    valid &= np.isfinite(ba) & (ba > 0.1) & (ba < 1.0)

    # Star-forming cut (default: include all, filter later by sSFR)
    sf_cut = None

    sfr = sfr[valid]
    mstar = mstar[valid]
    re_kpc = re_kpc[valid]
    z = z[valid]
    ba = ba[valid]
    n_gal = len(sfr)
    print(f"  Valid galaxies: {n_gal}")

    # Area = pi * R_e^2 (circularized)
    area_kpc2 = np.pi * re_kpc**2

    # SFR surface density
    sigma_sfr = sfr / area_kpc2  # M_sun/yr/kpc^2

    # Stellar mass surface density
    sigma_star = mstar / area_kpc2  # M_sun/kpc^2

    # Specific SFR
    ssfr = sfr / mstar  # yr^-1
    ssfr_gyr = ssfr * 1e9  # Gyr^-1

    # --- Molecular gas mass from Tacconi+2020 scaling ---
    # log(M_mol) = A + B*log(sSFR/Gyr^-1) + C*log(M*/1e10)
    log_mstar_10 = np.log10(mstar) - 10
    log_ssfr_gyr = np.log10(np.maximum(ssfr_gyr, 1e-6))
    log_mmol = TACCONI_A + TACCONI_B * log_ssfr_gyr + TACCONI_C * log_mstar_10
    mmol = 10**log_mmol  # M_sun
    sigma_mol = mmol / area_kpc2  # M_sun/kpc^2

    # Safety: gas surface density can't be below stellar
    sigma_gas = np.maximum(sigma_mol, sigma_star * 0.01)

    # Inclination correction for SFR (rough: deproject by 1/cos(i))
    # cos(i) = sqrt((1-b/a^2)/(1-0.2^2)) for oblate spheroid with intrinsic q0=0.2
    # but for integrated SFR this is less important
    # Skip detailed inclination correction for integrated values

    print(f"\n  Σ_SFR range: [{np.log10(sigma_sfr.min()):.2f}, {np.log10(sigma_sfr.max()):.2f}]")
    print(f"  Σ_gas range: [{np.log10(sigma_gas.min()):.2f}, {np.log10(sigma_gas.max()):.2f}]")
    print(f"  Σ* range:    [{np.log10(sigma_star.min()):.2f}, {np.log10(sigma_star.max()):.2f}]")

    # Population split
    ssfr = sfr / mstar
    is_sf = ssfr > 1e-11
    n_sf = is_sf.sum()
    n_q = (~is_sf).sum()
    print(f"\n  Population: {n_sf} star-forming ({100*n_sf/n_gal:.0f}%), "
          f"{n_q} quiescent ({100*n_q/n_gal:.0f}%)")

    return {
        "sigma_sfr": sigma_sfr,
        "sigma_gas": sigma_gas,
        "sigma_star": sigma_star,
        "mstar": mstar,
        "re_kpc": re_kpc,
        "z": z,
        "sfr": sfr,
        "ssfr": ssfr,
        "ba": ba,
        "is_sf": is_sf,
    }


def fit_ks_law(data, mask=None):
    """Fit standard Kennicutt-Schmidt power law: log Σ_SFR = α + β log Σ_gas."""
    if mask is None:
        mask = np.ones(len(data["sigma_sfr"]), dtype=bool)
    log_sfr = np.log10(data["sigma_sfr"][mask])
    log_gas = np.log10(data["sigma_gas"][mask])

    def ks(x, a, b):
        return a + b * x

    popt, pcov = curve_fit(ks, log_gas, log_sfr, p0=[-3, 1.0])
    perr = np.sqrt(np.diag(pcov))
    resid = log_sfr - ks(log_gas, *popt)
    rms = np.sqrt(np.mean(resid**2))

    print(f"\n  Standard KS fit:")
    print(f"    α (intercept) = {popt[0]:.3f} ± {perr[0]:.3f}")
    print(f"    β (slope)     = {popt[1]:.3f} ± {perr[1]:.3f}")
    print(f"    RMS           = {rms:.4f} dex")
    print(f"    N             = {len(log_gas)}")

    return popt, perr, rms


def fit_symbolic_regression(data, mask=None, n_cores=12, label=""):
    """Discover star formation law form using PySR."""
    if mask is None:
        mask = np.ones(len(data["sigma_sfr"]), dtype=bool)
    log_sfr = np.log10(data["sigma_sfr"][mask])
    log_gas = np.log10(data["sigma_gas"][mask])

    print(f"\n  Running PySR on {len(log_gas)} galaxies {label}({n_cores} cores)...")
    sys.stdout.flush()

    from pysr import PySRRegressor

    niter = 30 if len(log_gas) > 5000 else 40
    model = PySRRegressor(
        niterations=niter,
        populations=8,
        population_size=50,
        ncyclesperiteration=300,
        procs=n_cores,
        multithreading=n_cores > 1,
        maxsize=25,
        parsimony=1e-4,
        warm_start=False,
        turbo=True,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "exp",
            "log",
            "sqrt",
            "square",
            "cube",
            "neg",
        ],
        model_selection="accuracy",
        early_stop_condition=1e-10,
        timeout_in_seconds=600,
        verbosity=0,
    )

    t0 = time.time()
    model.fit(log_gas[:, np.newaxis], log_sfr)
    elapsed = time.time() - t0
    print(f"  PySR complete in {elapsed:.0f}s")

    print(f"\n  Top equations:")
    for i, eq in enumerate(model.equations_.head(10).itertuples()):
        print(f"    {i+1}. {eq.equation}  (loss={eq.loss:.6f})")

    return model


def plot_results(data, ks_all, ks_sf, model_all, model_sf):
    """Plot star formation law with KS fit and PySR."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for idx, (mask, label, color, ks, psr) in enumerate([
        (slice(None), "All galaxies", "Blues", ks_all, model_all),
        (data.get("is_sf", slice(None)), "Star-forming only", "Greens", ks_sf, model_sf),
    ]):
        if isinstance(mask, slice):
            log_sfr = np.log10(data["sigma_sfr"])
            log_gas = np.log10(data["sigma_gas"])
        else:
            log_sfr = np.log10(data["sigma_sfr"][mask])
            log_gas = np.log10(data["sigma_gas"][mask])

        ax = axes[idx, 0]
        hb = ax.hexbin(log_gas, log_sfr, gridsize=60, cmap=color,
                       mincnt=1, bins="log", alpha=0.8)
        plt.colorbar(hb, ax=ax, label="Galaxies per bin (log)")

        x_grid = np.linspace(log_gas.min(), log_gas.max(), 200)
        # KS fit
        ax.plot(x_grid, ks[0] + ks[1] * x_grid, "r-", lw=2.5,
                label=f"KS: n={ks[1]:.2f}±{ks[3]:.2f}")
        # Kennicutt (1998)
        ax.plot(x_grid, -3.99 + 1.40 * x_grid, "k--", lw=1, alpha=0.5,
                label="Kennicutt 1998")
        # PySR best
        if psr is not None and len(psr.equations_) > 0:
            pred = psr.predict(log_gas[:, np.newaxis])
            sidx = np.argsort(log_gas)
            ax.plot(log_gas[sidx], pred[sidx], "g-", lw=2, alpha=0.7,
                    label=f"PySR: loss={psr.equations_.iloc[0].loss:.4f}")

        ax.set_xlabel("log Σ$_{\\mathrm{gas}}$ [M☉/kpc²]")
        ax.set_ylabel("log Σ$_{\\mathrm{SFR}}$ [M☉/yr/kpc²]")
        ax.set_title(f"({chr(97+idx)}) KS Law — {label}")
        ax.legend(fontsize=8)

        # Residuals
        ax = axes[idx, 1]
        resid = log_sfr - (ks[0] + ks[1] * log_gas)
        hb2 = ax.hexbin(log_gas, resid, gridsize=50, cmap="RdBu_r", mincnt=1)
        plt.colorbar(hb2, ax=ax, label="Count")
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.axhline(np.std(resid), color="r", ls=":", lw=0.7, alpha=0.5)
        ax.axhline(-np.std(resid), color="r", ls=":", lw=0.7, alpha=0.5)
        ax.set_xlabel("log Σ$_{\\mathrm{gas}}$ [M☉/kpc²]")
        ax.set_ylabel("Residual (dex)")
        ax.set_title(f"({chr(99+idx)}) KS Residuals — σ={np.std(resid):.3f} dex")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/sf_law.pdf", dpi=200)
    plt.savefig(f"{OUTDIR}/sf_law.png", dpi=150)
    print(f"\n  Saved {OUTDIR}/sf_law.pdf")
    plt.close()

    # --- Panel 3: Σ_SFR vs Σ* ---
    fig, ax = plt.subplots(figsize=(7, 5.5))
    log_sstar = np.log10(data["sigma_star"])
    log_sfr_all = np.log10(data["sigma_sfr"])
    hb = ax.hexbin(log_sstar, log_sfr_all, gridsize=60, cmap="Oranges",
                   mincnt=1, bins="log", alpha=0.8)
    plt.colorbar(hb, ax=ax, label="Galaxies per bin (log)")
    # KS fit to Σ_SFR vs Σ*
    v = np.isfinite(log_sstar) & np.isfinite(log_sfr_all)
    popt_sstar, _ = curve_fit(lambda x, a, b: a + b * x,
                              log_sstar[v], log_sfr_all[v], p0=[-3, 1])
    ax.plot(np.sort(log_sstar[v]),
            popt_sstar[0] + popt_sstar[1] * np.sort(log_sstar[v]),
            "r-", lw=2, label=f"slope={popt_sstar[1]:.2f}")
    ax.set_xlabel("log Σ* [M☉/kpc²]")
    ax.set_ylabel("log Σ$_{\\mathrm{SFR}}$ [M☉/yr/kpc²]")
    ax.set_title("(e) Σ$_{\\mathrm{SFR}}$ vs Σ*")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/sf_ms.pdf", dpi=200)
    plt.savefig(f"{OUTDIR}/sf_ms.png", dpi=150)
    print(f"  Saved {OUTDIR}/sf_ms.png")
    plt.close()


def main():
    print("=" * 60)
    print("Star Formation Law Discovery (MaNGA + PySR)")
    print("=" * 60)

    download_catalogs()
    result = extract_sf_law()
    if result is None:
        print("\n  Extraction failed — columns may differ from expected.")
        print("  Check DAPall documentation for current column names.")
        return

    n_cores = int(os.environ.get("N_CORES", "12"))

    # Fit all galaxies
    ks_all, ks_all_err, ks_all_rms = fit_ks_law(result)
    ks_all_params = (ks_all[0], ks_all[1], ks_all_err[0], ks_all_err[1], ks_all_rms)

    model_all = fit_symbolic_regression(result, n_cores=n_cores, label="(all) ")

    # Fit star-forming only
    mask_sf = result.get("is_sf", np.ones(len(result["sigma_sfr"]), dtype=bool))
    n_sf = mask_sf.sum()
    ks_sf, ks_sf_err, ks_sf_rms = (ks_all, ks_all_err, ks_all_rms) if n_sf < 100 else \
        fit_ks_law(result, mask=mask_sf)
    ks_sf_params = (ks_sf[0], ks_sf[1], ks_sf_err[0], ks_sf_err[1], ks_sf_rms)

    model_sf = None
    if n_sf >= 100:
        model_sf = fit_symbolic_regression(result, mask=mask_sf, n_cores=n_cores,
                                           label="(SF) ")

    # Plot
    plot_results(result, ks_all_params, ks_sf_params, model_all, model_sf)

    # Save numerical results
    def serialize_model(m):
        if m is None or len(m.equations_) == 0:
            return None
        return {
            "best_loss": float(m.equations_.iloc[0].loss),
            "best_equation": str(m.equations_.iloc[0].equation),
            "top5": [
                {"equation": str(row.equation), "loss": float(row.loss)}
                for row in m.equations_.head(5).itertuples()
            ],
        }

    results = {
        "n_galaxies": len(result["sigma_sfr"]),
        "n_star_forming": int(n_sf),
        "all": {
            "ks_alpha": float(ks_all[0]),
            "ks_alpha_err": float(ks_all_err[0]),
            "ks_beta": float(ks_all[1]),
            "ks_beta_err": float(ks_all_err[1]),
            "ks_rms": float(ks_all_rms),
            "pysr": serialize_model(model_all),
        },
        "star_forming": {
            "ks_alpha": float(ks_sf[0]),
            "ks_alpha_err": float(ks_sf_err[0]),
            "ks_beta": float(ks_sf[1]),
            "ks_beta_err": float(ks_sf_err[1]),
            "ks_rms": float(ks_sf_rms),
            "pysr": serialize_model(model_sf),
        },
        "sigma_sfr_range": [float(np.log10(result["sigma_sfr"].min())),
                            float(np.log10(result["sigma_sfr"].max()))],
        "sigma_gas_range": [float(np.log10(result["sigma_gas"].min())),
                            float(np.log10(result["sigma_gas"].max()))],
    }

    with open(f"{OUTDIR}/sf_law_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {OUTDIR}/sf_law_results.json")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""xCOLD GASS Star Formation Law: Σ_SFR vs Σ_gas from CO-detected galaxies.

Loads the xCOLD GASS catalog FITS file, computes surface densities,
fits the standard Kennicutt-Schmidt law, and discovers the functional
form using PySR symbolic regression.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os, sys, warnings, json, time
warnings.filterwarnings("ignore")

OUTDIR = "analysis/sf_law"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), OUTDIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

XCOLD_PATH = os.path.join(OUTPUT_DIR, "xcold_gass.fits")


def load_xcold():
    """Load xCOLD GASS catalog, handling CDS '---' null values."""
    from astropy.io import fits
    import io

    fixed_path = XCOLD_PATH.replace('.fits', '_fixed.fits')
    if not os.path.exists(fixed_path):
        if not os.path.exists(XCOLD_PATH):
            print(f"  ERROR: {XCOLD_PATH} not found.")
            return None, None
        # Fix the FITS file: replace '---' strings with NaN
        with open(XCOLD_PATH, 'rb') as f:
            raw = f.read()
        # Replace all occurrences of '---' (with surrounding spaces/formatting)
        import re
        # The ASCII columns have --- right-justified within their field width
        raw_fixed = raw.replace(b'---', b'NaN')
        # Write clean version
        with open(fixed_path, 'wb') as f:
            f.write(raw_fixed)
        print(f"  Fixed FITS written to {fixed_path}")

    hdul = fits.open(fixed_path)
    cols = hdul[1].columns.names
    data = hdul[1].data
    hdul.close()
    print(f"  Loaded {len(data)} entries")
    print(f"  Columns: {len(cols)}")
    return data, cols


def extract_sf_law():
    """Extract Sigma_SFR and Sigma_gas from xCOLD GASS catalog."""

    data, cols = load_xcold()
    if data is None:
        return None

    # Column name mapping (xCOLD GASS naming conventions)
    col_map = {
        "logMstar": "logMstar",
        "logSFR": "logSFR-BEST",
        "R50kpc": "R50kpc",
        "logMH2": "logMH2",
        "e_logMH2": "e_logMH2",
        "Flag_CO": "Flag-CO",
        "z": "zSDSS",
        "incl": "Incl",
    }

    # Verify all columns exist
    for key, name in col_map.items():
        if name not in cols:
            print(f"  ERROR: Column '{name}' not found in catalog!")
            print(f"  Available columns: {cols}")
            return None
        print(f"    {key:12s} -> {name}")

    # Extract columns (handle '---' null values by loading raw bytes)
    def safe_float(arr):
        raw = np.array(arr, dtype=str)
        raw = np.char.strip(raw)
        raw = np.where(raw == "---", "nan", raw)
        return raw.astype(float)

    logMstar = safe_float(data["logMstar"])
    logSFR = safe_float(data["logSFR-BEST"])
    R50kpc = safe_float(data["R50kpc"])
    logMH2 = safe_float(data["logMH2"])
    e_logMH2 = safe_float(data["e_logMH2"])
    flag_co = np.array(data["Flag-CO"], dtype=int).astype(float)
    z = safe_float(data["zSDSS"])
    incl = safe_float(data["Incl"])

    n_total = len(logMstar)
    print(f"\n  Total entries: {n_total}")

    # Quality cuts
    valid = np.ones(n_total, dtype=bool)

    # Flag-CO == 1 (secure CO detection)
    valid &= np.isfinite(flag_co) & (flag_co == 1)
    n_co = valid.sum()
    print(f"  Flag-CO == 1: {n_co}")

    # Finite values on key quantities
    for arr, name in [(logMstar, "logMstar"), (logSFR, "logSFR-BEST"),
                       (R50kpc, "R50kpc"), (logMH2, "logMH2"),
                       (e_logMH2, "e_logMH2")]:
        valid &= np.isfinite(arr)
    n_finite = valid.sum()
    print(f"  Finite values: {n_finite}")

    # logMH2 > 0
    valid &= logMH2 > 0
    n_mol = valid.sum()
    print(f"  logMH2 > 0: {n_mol}")

    # z < 0.05
    valid &= np.isfinite(z) & (z < 0.05) & (z > 0.001)
    n_z = valid.sum()
    print(f"  z < 0.05: {n_z}")

    # logMstar > 9
    valid &= logMstar > 9
    n_mass = valid.sum()
    print(f"  logMstar > 9: {n_mass}")

    # Apply cuts
    logMstar = logMstar[valid]
    logSFR = logSFR[valid]
    R50kpc = R50kpc[valid]
    logMH2 = logMH2[valid]
    e_logMH2 = e_logMH2[valid]
    z = z[valid]
    incl = incl[valid]
    n_gal = len(logMstar)
    print(f"\n  Final sample: {n_gal} galaxies")

    # Compute surface densities
    area_kpc2 = np.pi * R50kpc**2
    SFR = 10**logSFR
    MH2 = 10**logMH2

    sigma_sfr = SFR / area_kpc2       # M_sun/yr/kpc^2
    sigma_gas = MH2 / area_kpc2        # M_sun/kpc^2

    print(f"\n  Σ_SFR range: [{np.log10(sigma_sfr.min()):.3f}, {np.log10(sigma_sfr.max()):.3f}]")
    print(f"  Σ_gas range: [{np.log10(sigma_gas.min()):.3f}, {np.log10(sigma_gas.max()):.3f}]")
    print(f"  logM* range: [{logMstar.min():.2f}, {logMstar.max():.2f}]")
    print(f"  z range:     [{z.min():.4f}, {z.max():.4f}]")

    return {
        "sigma_sfr": sigma_sfr,
        "sigma_gas": sigma_gas,
        "logMstar": logMstar,
        "R50kpc": R50kpc,
        "logMH2": logMH2,
        "e_logMH2": e_logMH2,
        "z": z,
        "incl": incl,
        "logSFR": logSFR,
    }


def fit_ks_law(data):
    """Fit standard Kennicutt-Schmidt power law: log Σ_SFR = α + β log Σ_gas."""
    log_sfr = np.log10(data["sigma_sfr"])
    log_gas = np.log10(data["sigma_gas"])

    def ks(x, a, b):
        return a + b * x

    popt, pcov = curve_fit(ks, log_gas, log_sfr, p0=[-3.5, 1.0])
    perr = np.sqrt(np.diag(pcov))
    resid = log_sfr - ks(log_gas, *popt)
    rms = np.sqrt(np.mean(resid**2))

    print(f"\n  Standard KS fit:")
    print(f"    α (intercept) = {popt[0]:.3f} ± {perr[0]:.3f}")
    print(f"    β (slope)     = {popt[1]:.3f} ± {perr[1]:.3f}")
    print(f"    RMS           = {rms:.4f} dex")
    print(f"    N             = {len(log_gas)}")

    return popt, perr, rms


def fit_symbolic_regression(data, n_cores=12):
    """Discover star formation law form using PySR."""
    log_sfr = np.log10(data["sigma_sfr"])
    log_gas = np.log10(data["sigma_gas"])

    print(f"\n  Running PySR on {len(log_gas)} galaxies ({n_cores} cores, 30 iter)...")
    sys.stdout.flush()

    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=30,
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


def plot_results(data, ks_params, model):
    """Create hexbin plot of log Σ_SFR vs log Σ_gas with KS and PySR fits."""
    log_sfr = np.log10(data["sigma_sfr"])
    log_gas = np.log10(data["sigma_gas"])

    fig, ax = plt.subplots(figsize=(8, 6.5))

    hb = ax.hexbin(log_gas, log_sfr, gridsize=40, cmap="Blues",
                   mincnt=1, bins="log", alpha=0.85)
    cb = plt.colorbar(hb, ax=ax, label="Galaxies per bin (log)")
    cb.ax.tick_params(labelsize=8)

    x_grid = np.linspace(log_gas.min(), log_gas.max(), 300)

    # KS fit
    ks_alpha, ks_beta = ks_params[0], ks_params[1]
    ax.plot(x_grid, ks_alpha + ks_beta * x_grid, "r-", lw=2.5,
            label=f"KS law: log Σ_SFR = {ks_alpha:.2f} + {ks_beta:.2f} log Σ_gas")

    # Kennicutt (1998) canonical
    ax.plot(x_grid, -3.99 + 1.40 * x_grid, "k--", lw=1, alpha=0.5,
            label="Kennicutt 1998 (n=1.40)")

    # PySR best fit
    if model is not None and len(model.equations_) > 0:
        pred = model.predict(log_gas[:, np.newaxis])
        sidx = np.argsort(log_gas)
        ax.plot(log_gas[sidx], pred[sidx], "g-", lw=2, alpha=0.7,
                label=f"PySR: loss={model.equations_.iloc[0].loss:.4f}")

    ax.set_xlabel("log Σ$_{\\mathrm{mol}}$ [M$_\\odot$ kpc$^{-2}$]", fontsize=12)
    ax.set_ylabel("log Σ$_{\\mathrm{SFR}}$ [M$_\\odot$ yr$^{-1}$ kpc$^{-2}$]", fontsize=12)
    ax.set_title(f"xCOLD GASS Star Formation Law ({len(log_sfr)} galaxies)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")

    # Annotate KS parameters
    textstr = f"KS: α={ks_alpha:.3f}±{ks_params[2]:.3f}\n" \
              f"     β={ks_beta:.3f}±{ks_params[3]:.3f}\n" \
              f"RMS={ks_params[4]:.4f} dex"
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85)
    ax.text(0.04, 0.96, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, "xcold_gass.pdf")
    plt.savefig(pdf_path, dpi=200)
    print(f"  Saved {pdf_path}")
    plt.close()


def main():
    print("=" * 60)
    print("xCOLD GASS Star Formation Law (PySR + KS)")
    print("=" * 60)

    result = extract_sf_law()
    if result is None:
        print("\n  Extraction failed — check FITS file availability and columns.")
        return

    n_cores = int(os.environ.get("N_CORES", "12"))

    # Standard KS fit
    ks_all, ks_all_err, ks_all_rms = fit_ks_law(result)
    ks_params = (ks_all[0], ks_all[1], ks_all_err[0], ks_all_err[1], ks_all_rms)

    # PySR symbolic regression
    model = fit_symbolic_regression(result, n_cores=n_cores)

    # Plot
    plot_results(result, ks_params, model)

    # Serialize PySR results
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

    # Save results
    results = {
        "n_galaxies": len(result["sigma_sfr"]),
        "ks_alpha": float(ks_all[0]),
        "ks_alpha_err": float(ks_all_err[0]),
        "ks_beta": float(ks_all[1]),
        "ks_beta_err": float(ks_all_err[1]),
        "ks_rms": float(ks_all_rms),
        "pysr": serialize_model(model),
        "sigma_sfr_range": [float(np.log10(result["sigma_sfr"].min())),
                            float(np.log10(result["sigma_sfr"].max()))],
        "sigma_gas_range": [float(np.log10(result["sigma_gas"].min())),
                            float(np.log10(result["sigma_gas"].max()))],
    }

    json_path = os.path.join(OUTPUT_DIR, "xcold_gass_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Phase 3: MaNGA cross-validation using DAP summary catalog.

Downloads dapall catalog, extracts V_rot and M* at 1Re,
computes approximate g_bar and g_obs, compares with SPARC CPX5 prediction.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os, sys, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/phase3"
os.makedirs(OUTDIR, exist_ok=True)

CATALOG_URL = "https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/dapall-v3_1_1-3.1.0.fits"
CATALOG_PATH = os.path.join(OUTDIR, "manga_dapall.fits")

CPX5_A = -17.06
CPX5_B = -72.71

G_SI = 6.6743e-11
Msun_kg = 1.989e30
kpc_m = 3.0857e19


def download_catalog():
    """Download MaNGA DAPall catalog if not present."""
    if os.path.exists(CATALOG_PATH):
        print(f"  Using cached {CATALOG_PATH}")
        return

    print(f"  Downloading MaNGA DAPall catalog (146 MB)...")
    import urllib.request
    urllib.request.urlretrieve(CATALOG_URL, CATALOG_PATH)
    print(f"  Downloaded.")


def extract_rar_points():
    """Extract approximate (g_bar, g_obs) from MaNGA DAPall."""
    try:
        from astropy.io import fits
    except ImportError:
        print("  Installing astropy...")
        import subprocess
        subprocess.check_call(["pip3", "install", "astropy", "--break-system-packages", "-q"])
        from astropy.io import fits

    hdul = fits.open(CATALOG_PATH)
    data = hdul[1].data  # extension 1 has the per-galaxy data
    cols = hdul[1].columns
    print(f"  Columns available: {len(cols.names)}")

    # Find relevant columns
    col_names = [c.lower() for c in cols.names]
    print(f"  Searching for velocity/stellar mass columns...")

    # Key columns we need (from DAP documentation):
    # - stellar_vel: stellar velocity at effective radius
    # - nsa_sersic_ba: axis ratio (for inclination correction)
    # - nsa_elpetro_mass: stellar mass
    # - nsa_sersic_th50: effective radius

    # Print relevant column matches
    for pattern in ["vel", "mass", "sersic", "re", "nsa"]:
        matches = [c for c in cols.names if pattern in c.lower()]
        if matches:
            print(f"    '{pattern}' → {matches[:5]}")

    # Extract key quantities
    results = {}

    # Stellar mass
    for col in cols.names:
        if "nsa_elpetro_mass" in col.lower():
            results["mstar"] = data[col]
            print(f"    Stellar mass: {col}")
            break

    # Effective radius (arcsec)
    for col in cols.names:
        if "nsa_sersic_th50" in col.lower():
            results["re_arcsec"] = data[col]
            print(f"    Re (arcsec): {col}")
            break

    # Axis ratio
    for col in cols.names:
        if "nsa_sersic_ba" in col.lower():
            results["ba"] = data[col]
            print(f"    b/a: {col}")
            break

    # Redshift
    for col in cols.names:
        if col.lower() == "nsa_z" or "nsa_redshift" in col.lower():
            results["z"] = data[col]
            print(f"    Redshift: {col}")
            break

    # Velocity dispersion at Re
    for col in cols.names:
        if "sigma" in col.lower() and "re" in col.lower():
            results["sigma_re"] = data[col]
            print(f"    sigma(Re): {col}")
            break

    # Rotational velocity
    for col in cols.names:
        if "vrot" in col.lower() or "rot" in col.lower():
            results["vrot"] = data[col]
            print(f"    V_rot: {col}")
            break

    hdul.close()

    if "mstar" not in results:
        print("  Could not find stellar mass column. Using alternative approach.")
        return None

    n_gal = len(results["mstar"])
    print(f"\n  Extracted data for {n_gal} galaxies")

    # Compute g_bar and g_obs for each galaxy
    valid_mask = np.ones(n_gal, dtype=bool)

    mstar = results["mstar"].astype(float)
    valid_mask &= np.isfinite(mstar) & (mstar > 1e7)

    # Convert Re from arcsec to kpc using redshift
    if "z" in results and "re_arcsec" in results:
        z = np.maximum(results["z"].astype(float), 0.001)
        da = z * 299792.458 / 70.0 * 1000  # approximate angular diameter distance in kpc/arcsec
        re_kpc = results["re_arcsec"].astype(float) * da / 206265  # arcsec → kpc
        valid_mask &= np.isfinite(re_kpc) & (re_kpc > 0) & (re_kpc < 100)
    else:
        # Assume typical Re ~ 3 kpc
        re_kpc = np.full(n_gal, 3.0)

    # g_bar ≈ G * M* / Re² (baryonic acceleration at effective radius)
    g_bar = G_SI * mstar * Msun_kg / (re_kpc * kpc_m)**2
    valid_mask &= np.isfinite(g_bar) & (g_bar > 1e-14)

    # g_obs from V_rot²/Re if available, else from sigma
    if "vrot" in results:
        vrot = results["vrot"].astype(float)
        valid_mask &= np.isfinite(vrot) & (vrot > 0) & (vrot < 1000)
        g_obs = (vrot * 1000)**2 / (re_kpc * kpc_m)  # km/s → m/s
    elif "sigma_re" in results:
        sigma_re = results["sigma_re"].astype(float)
        valid_mask &= np.isfinite(sigma_re) & (sigma_re > 10) & (sigma_re < 500)
        g_obs = 3 * (sigma_re * 1000)**2 / (re_kpc * kpc_m)  # V_rot ≈ √3 σ for dispersion-supported
    else:
        print("  No velocity column found.")
        return None

    valid_mask &= np.isfinite(g_obs) & (g_obs > 1e-14)

    g_bar = g_bar[valid_mask]
    g_obs = g_obs[valid_mask]
    mstar = mstar[valid_mask]

    print(f"  Valid galaxies: {len(g_bar)}")
    print(f"  log g_bar range: [{np.log10(g_bar.min()):.2f}, {np.log10(g_bar.max()):.2f}]")
    print(f"  log g_obs range: [{np.log10(g_obs.min()):.2f}, {np.log10(g_obs.max()):.2f}]")

    return np.log10(g_bar), np.log10(g_obs), mstar


def compare_with_sparc(outdir=OUTDIR):
    """Download MaNGA catalog and compare RAR with SPARC CPX5."""
    print("=" * 60)
    print("Phase 3: MaNGA Cross-Validation")
    print("=" * 60)

    download_catalog()

    result = extract_rar_points()
    if result is None:
        print("\n  Could not extract RAR points from MaNGA catalog.")
        print("  The DAPall format may differ from expected.")
        print("  Alternative: use SDSS CasJobs to query specific columns.")
        return

    log_gbar, log_gobs, mstar = result

    # Fit CPX5 to MaNGA
    def cpx5_log(x, a, b):
        return a + b / np.maximum(x, -50)

    popt, pcov = curve_fit(cpx5_log, log_gbar, log_gobs, p0=[-17, -70], maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    pred = cpx5_log(log_gbar, *popt)
    rms = np.sqrt(np.mean((log_gobs - pred)**2))

    print(f"\n  MaNGA CPX5 fit:")
    print(f"    a = {popt[0]:.2f} ± {perr[0]:.2f}")
    print(f"    b = {popt[1]:.2f} ± {perr[1]:.2f}")
    print(f"    RMS = {rms:.4f} dex")

    # Compare with SPARC
    da = popt[0] - CPX5_A
    db = popt[1] - CPX5_B
    print(f"\n  Comparison with SPARC global fit (a={CPX5_A}, b={CPX5_B}):")
    print(f"    Δa = {da:+.2f}, Δb = {db:+.0f}")

    # Compare with SPARC MASSIVE sub-sample
    sparc_massive_a = -16.78
    sparc_massive_b = -70.29
    print(f"    SPARC massive (a={sparc_massive_a}, b={sparc_massive_b}):")
    print(f"    Δa = {popt[0]-sparc_massive_a:+.2f}, Δb = {popt[1]-sparc_massive_b:+.0f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    # Hexbin for 10k+ galaxies
    hb = ax.hexbin(log_gbar, log_gobs, gridsize=50, cmap="Blues",
                    mincnt=1, bins="log")
    x_grid = np.linspace(log_gbar.min(), log_gbar.max(), 200)
    ax.plot(x_grid, cpx5_log(x_grid, *popt), "r-", lw=2.5,
            label=f"MaNGA CPX5: a={popt[0]:.2f}, b={popt[1]:.0f}")
    ax.plot(x_grid, cpx5_log(x_grid, CPX5_A, CPX5_B), "k--", lw=1.5, alpha=0.7,
            label=f"SPARC CPX5: a={CPX5_A:.2f}, b={CPX5_B:.0f}")
    ax.plot(x_grid, x_grid, "k:", lw=0.5, alpha=0.3)
    plt.colorbar(hb, ax=ax, label="Galaxies per bin")
    ax.set_xlabel("log g_bar [m/s²]")
    ax.set_ylabel("log g_obs [m/s²]")
    ax.set_title("(a) MaNGA RAR (DAPall catalog)")
    ax.legend(fontsize=8)

    ax = axes[1]
    # SPARC CPX5 residuals for MaNGA
    resid = log_gobs - cpx5_log(log_gbar, CPX5_A, CPX5_B)
    ax.hexbin(log_gbar, resid, gridsize=40, cmap="RdBu_r", mincnt=1)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("log g_bar [m/s²]")
    ax.set_ylabel("Residual from SPARC CPX5 (dex)")
    ax.set_title(f"(b) MaNGA − SPARC CPX5 Residuals (mean={np.mean(resid):+.3f})")
    plt.colorbar(ax.collections[0], ax=ax, label="Count")

    plt.tight_layout()
    plt.savefig(f"{outdir}/manga_rar.pdf", dpi=200)
    plt.savefig(f"{outdir}/manga_rar.png", dpi=150)
    print(f"\n  Saved {outdir}/manga_rar.png")
    plt.close()

    return popt, perr


if __name__ == "__main__":
    compare_with_sparc()
    print("\nDone.")

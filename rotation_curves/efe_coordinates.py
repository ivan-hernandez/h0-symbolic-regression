"""Proper EFE analysis: cross-match SPARC galaxies with SIMBAD for RA/Dec.

Computes 3D isolation (distance in RA/Dec/D space) and checks if RAR
residuals correlate with environmental density — MOND's key prediction.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parse_sparc import parse_mass_models, compute_radial_accelerations
import time, json, os

# ── SIMBAD query ──────────────────────────────────────────────────────────────

def query_simbad_coords(galaxy_names, cache_path="galaxy_coords.json"):
    """Query SIMBAD for RA/Dec of SPARC galaxies, with caching."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    from astroquery.simbad import Simbad
    coords = {}
    for i, name in enumerate(galaxy_names):
        try:
            result = Simbad.query_object(name)
            if result is not None and len(result) > 0:
                ra = result['ra'][0]
                dec = result['dec'][0]
                coords[name] = {"ra": str(ra), "dec": str(dec)}
            else:
                # Try without leading zeros
                clean = name.lstrip('0')
                if clean != name:
                    result = Simbad.query_object(clean)
                    if result is not None and len(result) > 0:
                        ra = result['ra'][0]
                        dec = result['dec'][0]
                        coords[name] = {"ra": str(ra), "dec": str(dec)}
                    else:
                        print(f"  WARNING: {name} not found in SIMBAD")
                else:
                    print(f"  WARNING: {name} not found in SIMBAD")
        except Exception as e:
            print(f"  ERROR querying {name}: {e}")
        if (i+1) % 10 == 0:
            print(f"  Queried {i+1}/{len(galaxy_names)} galaxies...")
        time.sleep(0.3)  # Be nice to SIMBAD

    with open(cache_path, "w") as f:
        json.dump(coords, f, indent=2)
    print(f"  Cached {len(coords)} coordinates to {cache_path}")
    return coords


def parse_angle(angle_str):
    """Parse to decimal degrees. SIMBAD now returns decimal directly."""
    try:
        val = float(angle_str)
        return val
    except (ValueError, TypeError):
        pass
    # Fallback: HH:MM:SS.S
    try:
        parts = str(angle_str).split(":")
        if len(parts) == 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return (h + m/60 + s/3600) * 15
    except:
        pass
    return np.nan


# ── 3D isolation ──────────────────────────────────────────────────────────────

def compute_3d_separation(ra1_deg, dec1_deg, d1_mpc, ra2_deg, dec2_deg, d2_mpc):
    """Compute 3D distance between two galaxies given RA, Dec, and distance."""
    ra1_r, dec1_r = np.radians(ra1_deg), np.radians(dec1_deg)
    ra2_r, dec2_r = np.radians(ra2_deg), np.radians(dec2_deg)

    # Angular separation
    cos_sep = (np.sin(dec1_r) * np.sin(dec2_r)
               + np.cos(dec1_r) * np.cos(dec2_r) * np.cos(ra1_r - ra2_r))
    cos_sep = np.clip(cos_sep, -1, 1)
    ang_sep_rad = np.arccos(cos_sep)

    # 3D distance via law of cosines
    sep_3d = np.sqrt(d1_mpc**2 + d2_mpc**2 - 2 * d1_mpc * d2_mpc * np.cos(ang_sep_rad))
    return sep_3d


def efe_3d_analysis(outdir="analysis"):
    """Main EFE analysis with 3D positions."""
    print("=" * 60)
    print("Proper EFE Analysis (3D isolation with SIMBAD coordinates)")
    print("=" * 60)

    df = parse_mass_models()
    acc = compute_radial_accelerations(df)

    # Filter valid
    valid = np.isfinite(acc["log_gbar"]) & np.isfinite(acc["log_gobs"]) & (acc["gbar"] > 0)
    acc = acc[valid].copy()
    print(f"  Valid points: {len(acc)}")

    # Get unique galaxies
    galaxies = acc["ID"].unique()
    print(f"  Unique galaxies: {len(galaxies)}")

    # Get galaxy properties
    gal_info = acc.groupby("ID").agg({
        "D": "first",
    }).to_dict("index")

    # Query coordinates (cached)
    coords = query_simbad_coords(galaxies.tolist())

    # Only galaxies with coordinates
    valid_gals = [g for g in galaxies if g in coords]
    print(f"  Galaxies with coordinates: {len(valid_gals)}")

    gal_ra = {}
    gal_dec = {}
    for g in valid_gals:
        ra_str = coords[g]["ra"]
        dec_str = coords[g]["dec"]
        ra_deg = parse_angle(ra_str)
        dec_deg = parse_angle(dec_str)
        # SIMBAD returns dec in degrees, but verify
        if dec_deg is not None and dec_deg != dec_deg:
            dec_deg = np.nan
        gal_ra[g] = ra_deg
        gal_dec[g] = dec_deg

    # Compute 3D nearest neighbor distance
    isolation_3d = {}
    n_valid = 0
    for g1 in valid_gals:
        if np.isnan(gal_ra[g1]) or np.isnan(gal_dec[g1]):
            continue
        n_valid += 1
        min_sep = np.inf
        for g2 in valid_gals:
            if g1 == g2:
                continue
            if np.isnan(gal_ra[g2]) or np.isnan(gal_dec[g2]):
                continue
            sep = compute_3d_separation(
                gal_ra[g1], gal_dec[g1], gal_info[g1]["D"],
                gal_ra[g2], gal_dec[g2], gal_info[g2]["D"]
            )
            if sep < min_sep:
                min_sep = sep
        isolation_3d[g1] = min_sep

    print(f"  Galaxies with valid 3D isolation: {n_valid}")

    # Fit CPX5 residuals
    def cpx5_log(x, a, b):
        return a + b / x
    popt, _ = curve_fit(cpx5_log, acc["log_gbar"].values, acc["log_gobs"].values,
                         p0=[-12, -50], maxfev=10000)
    acc["resid_CPX5"] = acc["log_gobs"] - cpx5_log(acc["log_gbar"].values, *popt)
    acc["isolation_3d"] = acc["ID"].map(isolation_3d)
    acc["log_isolation_3d"] = np.log10(np.maximum(acc["isolation_3d"], 0.1))

    # Gas fraction from original df
    df["gas_frac"] = np.abs(df["Vgas"]) / np.maximum(df["Vobs"], 0.1)
    gas_frac_map = df.groupby("ID")["gas_frac"].median().to_dict()
    acc["gas_frac"] = acc["ID"].map(gas_frac_map)

    # Correlation
    from scipy.stats import spearmanr
    mask_iso = np.isfinite(acc["log_isolation_3d"])
    r_iso, p_iso = spearmanr(acc.loc[mask_iso, "log_isolation_3d"],
                              acc.loc[mask_iso, "resid_CPX5"])
    r_gas, p_gas = spearmanr(acc["gas_frac"], acc["resid_CPX5"])
    print(f"\n  Spearman(3D Isolation, Resid): ρ={r_iso:.4f}, p={p_iso:.2e}")
    print(f"  Spearman(Gas frac, Resid): ρ={r_gas:.4f}, p={p_gas:.2e}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    scatter = ax.scatter(acc.loc[mask_iso, "log_isolation_3d"],
                         acc.loc[mask_iso, "resid_CPX5"],
                         s=2, alpha=0.3, c=acc.loc[mask_iso, "gas_frac"], cmap="viridis")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("log nearest neighbor 3D distance (Mpc)")
    ax.set_ylabel("CPX5 residual (dex)")
    plt.colorbar(scatter, ax=ax, label="gas fraction")

    # Binned
    log_iso = acc.loc[mask_iso, "log_isolation_3d"].values
    resid = acc.loc[mask_iso, "resid_CPX5"].values
    bins = np.linspace(log_iso.min(), log_iso.max(), 15)
    bin_c, bin_m, bin_s = [], [], []
    for i in range(len(bins)-1):
        m = (log_iso >= bins[i]) & (log_iso < bins[i+1])
        if m.sum() > 10:
            bin_c.append((bins[i]+bins[i+1])/2)
            bin_m.append(np.median(resid[m]))
            bin_s.append(np.std(resid[m])/np.sqrt(m.sum()))
    ax.errorbar(bin_c, bin_m, yerr=bin_s, fmt="r.-", lw=2, capsize=3)

    # Per-galaxy mean residual vs isolation
    ax = axes[0, 1]
    gal_resid = acc.groupby("ID")["resid_CPX5"].mean()
    gal_iso = acc.groupby("ID")["isolation_3d"].first()
    valid_gal = gal_resid.index.intersection(gal_iso.dropna().index)
    ax.scatter(np.log10(np.maximum(gal_iso[valid_gal], 0.1)),
               gal_resid[valid_gal], s=10, alpha=0.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("log 3D isolation (Mpc)")
    ax.set_ylabel("Mean CPX5 residual (dex)")

    # Histogram of isolation
    ax = axes[1, 0]
    ax.hist(isolation_3d.values(), bins=30)
    ax.set_xlabel("3D nearest neighbor distance (Mpc)")
    ax.set_ylabel("Count")

    # Per-galaxy RMS vs isolation
    ax = axes[1, 1]
    gal_rms = acc.groupby("ID")["resid_CPX5"].std()
    valid_gal2 = gal_rms.index.intersection(gal_iso.dropna().index)
    ax.scatter(np.log10(np.maximum(gal_iso[valid_gal2], 0.1)),
               gal_rms[valid_gal2], s=10, alpha=0.5)
    ax.set_xlabel("log 3D isolation (Mpc)")
    ax.set_ylabel("Per-galaxy RMS (dex)")

    plt.tight_layout()
    plt.savefig(f"{outdir}/efe_3d_analysis.png", dpi=150)
    print(f"  Saved {outdir}/efe_3d_analysis.png")
    plt.close()

    # Summary statistics
    iso_vals = np.array(list(isolation_3d.values()))
    print(f"\n  3D Isolation summary (Mpc):")
    print(f"    Median: {np.median(iso_vals):.1f} Mpc")
    print(f"    Range: [{np.min(iso_vals):.1f}, {np.max(iso_vals):.1f}]")
    print(f"    Galaxies with neighbor < 1 Mpc: {np.sum(iso_vals < 1)}")
    print(f"    Galaxies with neighbor < 5 Mpc: {np.sum(iso_vals < 5)}")
    print(f"    Galaxies with neighbor > 10 Mpc: {np.sum(iso_vals > 10)}")

    return acc, isolation_3d


if __name__ == "__main__":
    import sys
    efe_3d_analysis()

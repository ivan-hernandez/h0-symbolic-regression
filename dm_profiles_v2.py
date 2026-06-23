"""DM profiles v2: Parametric NFW/Einasto/Burkert fits to V(R).

Fixes from debate: replace noise-amplifying numerical derivatives with
direct parametric fits to observed rotation curves. For each galaxy,
compute V²(R) from NFW/Burkert/Einasto density profiles and fit.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from parse_sparc import parse_mass_models
import os, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/dm_profiles"
os.makedirs(OUTDIR, exist_ok=True)
kpc_to_m = 3.0857e19
G_SI = 6.6743e-11
Msun_kg = 1.989e30


def V_nfw(R_kpc, log_M200, log_c):
    """NFW rotation curve. R in kpc, returns V in km/s."""
    M200 = 10**log_M200
    c = 10**log_c
    rho_crit = 1.37e-7  # Msun/pc³
    R200 = (M200 / (200 * 4*np.pi/3 * rho_crit * 1e9))**(1/3)  # kpc
    rs = R200 / c
    rho_s = M200 / (4*np.pi*rs**3 * (np.log(1+c) - c/(1+c))) * Msun_kg / (kpc_to_m**3)

    x = R_kpc / rs
    M_enc = 4*np.pi*rho_s*rs**3 * (np.log(1+x) - x/(1+x))
    V2 = G_SI * M_enc / (R_kpc * kpc_to_m) / 1e6  # km²/s²
    return np.sqrt(np.maximum(V2, 0))


def V_burkert(R_kpc, log_rho0, log_r0):
    """Burkert (cored) rotation curve."""
    rho0 = 10**log_rho0  # Msun/kpc³
    r0 = 10**log_r0      # kpc
    x = R_kpc / r0
    M_enc = 2*np.pi*rho0*r0**3 * (np.log(1+x) + 0.5*np.log(1+x**2) - np.arctan(x))
    V2 = G_SI * M_enc / (R_kpc * kpc_to_m) / 1e6
    return np.sqrt(np.maximum(V2, 0))


def V_einasto(R_kpc, log_rho0, log_r0, alpha):
    """Einasto rotation curve (approximate enclosed mass)."""
    rho0 = 10**log_rho0
    r0 = 10**log_r0
    r = R_kpc / r0
    # Approximate M_enc for Einasto via gamma function
    n = 1/alpha
    from scipy.special import gammainc
    d_n = 3*n - 1/3 + 0.0079/n
    x = d_n * r**alpha
    gam = gammainc(3*n, x)
    M_enc = 4*np.pi*rho0*r0**3 * n * np.exp(d_n) * d_n**(-3*n) * gam
    V2 = G_SI * M_enc / (R_kpc * kpc_to_m) / 1e6
    return np.sqrt(np.maximum(V2, 0))


def fit_profiles(outdir=OUTDIR):
    print("=" * 60)
    print("DM Profiles v2 — Parametric Fits to V(R)")
    print("=" * 60)

    df = parse_mass_models()
    df = df[df["R"] > 0].copy()

    galaxies = df.groupby("ID").size()
    top_gals = galaxies[galaxies >= 8].index  # ≥8 points for stable fit

    results = []
    for gal in top_gals:
        sub = df[df["ID"] == gal].sort_values("R")
        R = sub["R"].values
        V_obs = sub["Vobs"].values
        e_V = sub["e_Vobs"].values

        if len(R) < 8:
            continue

        # Fit NFW
        try:
            popt_nfw, _ = curve_fit(V_nfw, R, V_obs,
                                     p0=[11.5, 1.0],
                                     sigma=np.maximum(e_V, 5),
                                     maxfev=10000, absolute_sigma=True)
            pred_nfw = V_nfw(R, *popt_nfw)
            chi2_nfw = np.sum(((V_obs - pred_nfw) / np.maximum(e_V, 5))**2)
            c_nfw = 10**popt_nfw[1]
        except:
            popt_nfw = [np.nan, np.nan]
            chi2_nfw = np.inf
            c_nfw = np.nan

        # Fit Burkert (core)
        try:
            popt_bur, _ = curve_fit(V_burkert, R, V_obs,
                                     p0=[6, 0.5],
                                     sigma=np.maximum(e_V, 5),
                                     maxfev=10000, absolute_sigma=True)
            pred_bur = V_burkert(R, *popt_bur)
            chi2_bur = np.sum(((V_obs - pred_bur) / np.maximum(e_V, 5))**2)
        except:
            popt_bur = [np.nan, np.nan]
            chi2_bur = np.inf

        # Prefer NFW or Burkert?
        k_nfw, k_bur = 2, 2
        aic_nfw = chi2_nfw + 2*k_nfw
        aic_bur = chi2_bur + 2*k_bur
        best = "NFW" if aic_nfw < aic_bur else "Burkert"

        V_max = V_obs.max()
        results.append({
            "galaxy": gal, "n_pts": len(R),
            "log_M200": popt_nfw[0], "log_c": popt_nfw[1], "c_nfw": c_nfw,
            "chi2_nfw": chi2_nfw, "chi2_bur": chi2_bur,
            "best": best, "V_max": V_max,
        })

    df_r = pd.DataFrame(results).dropna(subset=["c_nfw"])
    print(f"\n  Fitted {len(df_r)} galaxies")

    # cusp vs core
    n_nfw = (df_r["best"] == "NFW").sum()
    n_bur = (df_r["best"] == "Burkert").sum()
    print(f"  NFW (cusp) preferred: {n_nfw} ({100*n_nfw/len(df_r):.0f}%)")
    print(f"  Burkert (core) preferred: {n_bur} ({100*n_bur/len(df_r):.0f}%)")

    print(f"\n  NFW concentration: median c = {np.median(df_r['c_nfw']):.1f}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    # Show fits for a few galaxies
    for i, (_, row) in enumerate(df_r.iloc[:8].iterrows()):
        gal = row["galaxy"]
        sub = df[df["ID"] == gal].sort_values("R")
        ax.errorbar(sub["R"], sub["Vobs"], yerr=sub["e_Vobs"],
                     fmt="o", ms=3, capsize=1, alpha=0.5)
        if row["best"] == "NFW":
            R_grid = np.linspace(sub["R"].min(), sub["R"].max(), 100)
            ax.plot(R_grid, V_nfw(R_grid, row["log_M200"], row["log_c"]),
                    "r-", lw=1, alpha=0.5)
        else:
            R_grid = np.linspace(sub["R"].min(), sub["R"].max(), 100)
            # Re-fit for plot (just visual)
            try:
                p, _ = curve_fit(V_burkert, sub["R"], sub["Vobs"], p0=[6, 0.5], maxfev=5000)
                ax.plot(R_grid, V_burkert(R_grid, *p), "b-", lw=1, alpha=0.5)
            except:
                pass
    ax.set_xlabel("R [kpc]"); ax.set_ylabel("V_c [km/s]")
    ax.set_title("(a) Parametric Fits (NFW=red, Burkert=blue)")

    ax = axes[1]
    colors = ["red" if b == "NFW" else "blue" for b in df_r["best"]]
    ax.scatter(df_r["V_max"], df_r["c_nfw"], s=10, c=colors, alpha=0.6)
    ax.axhline(10, color="k", ls="--", lw=0.5, label="c=10 (MW-like)")
    ax.set_xlabel("V_max [km/s]"); ax.set_ylabel("NFW concentration c")
    ax.set_title(f"(b) Concentration vs Mass ({n_nfw} NFW, {n_bur} Burkert)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[2]
    counts = [n_nfw, n_bur]
    bars = ax.bar(["NFW (cusp)", "Burkert (core)"], counts,
                   color=["red", "blue"], edgecolor="k")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                str(count), ha="center", fontweight="bold")
    ax.set_ylabel("Number of galaxies")
    ax.set_title(f"(c) Preferred Model (ΔAIC, {len(df_r)} galaxies)")

    plt.tight_layout()
    plt.savefig(f"{outdir}/dm_profiles_v2.pdf", dpi=200)
    plt.savefig(f"{outdir}/dm_profiles_v2.png", dpi=150)
    print(f"\n  Saved {outdir}/dm_profiles_v2.png")
    plt.close()

    df_r.to_csv(f"{outdir}/dm_profiles_v2.csv", index=False)
    return df_r


if __name__ == "__main__":
    fit_profiles()
    print("\nDone.")

"""TFR v2: Fixed M_b, proper errors, split by acceleration regime.

Fixes from debate:
- M_b uses per-point V²(R_last)*R_last/G (not average V²)
- Errors use intrinsic scatter (0.30 dex) not fabricated 0.05 dex
- Split by g_bar regime to test MOND in deep-MOND only
- Bootstrap uncertainties on slope
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from parse_sparc import parse_mass_models
import os, warnings
warnings.filterwarnings("ignore")

OUTDIR = "analysis/tfr"
os.makedirs(OUTDIR, exist_ok=True)
kpc_to_m = 3.0857e19
G_SI = 6.6743e-11
Msun_kg = 1.989e30
RNG = np.random.RandomState(42)


def compute_properties():
    """Compute V_flat and M_b per galaxy — FIXED."""
    df = parse_mass_models()
    df = df[df["R"] > 0].copy()

    results = []
    for gal in df["ID"].unique():
        sub = df[df["ID"] == gal].sort_values("R")
        if len(sub) < 5:
            continue

        # V_flat: outer velocity (last 3 pts)
        v_outer = sub["Vobs"].values[-3:]
        V_flat = np.mean(v_outer)

        # V_max
        V_max = sub["Vobs"].max()

        # M_b at R_last using PER-POINT V²(R_last) * R_last / G
        R_last_kpc = sub["R"].values[-1]
        R_last_m = R_last_kpc * kpc_to_m

        V_c_last = sub["Vobs"].values[-1]
        V_gas_last = np.abs(sub["Vgas"].values[-1])
        V_disk_last = sub["Vdisk"].values[-1]
        V_bul_last = sub["Vbul"].values[-1]

        M_tot = V_c_last**2 * 1e6 * R_last_m / G_SI / Msun_kg
        M_gas = V_gas_last**2 * 1e6 * R_last_m / G_SI / Msun_kg
        M_star = (0.5*V_disk_last**2 + 0.7*V_bul_last**2) * 1e6 * R_last_m / G_SI / Msun_kg
        M_b = M_gas + M_star
        f_gas = M_gas / max(M_b, 1e-10)

        # g_bar at outer point (for regime classification)
        g_bar_last = (V_gas_last**2 + 0.5*V_disk_last**2 + 0.7*V_bul_last**2) * 1e6 / R_last_m
        g_obs_last = V_c_last**2 * 1e6 / R_last_m

        results.append({
            "galaxy": gal, "n_pts": len(sub),
            "V_flat": V_flat, "V_max": V_max,
            "M_b": M_b, "M_gas": M_gas, "M_star": M_star,
            "f_gas": f_gas,
            "g_bar_last": g_bar_last,
            "g_obs_last": g_obs_last,
            "D": sub["D"].iloc[0],
        })

    return pd.DataFrame(results)


def fit_tfr(df, outdir=OUTDIR):
    """Fit TFR with proper errors and bootstrap."""
    print("=" * 60)
    print("TFR v2 — Fixed M_b, Proper Errors")
    print("=" * 60)

    V = df["V_flat"].values
    M = df["M_b"].values
    g_bar = df["g_bar_last"].values
    log_V = np.log10(V)
    log_M = np.log10(np.maximum(M, 1e-10))
    log_gbar = np.log10(np.maximum(g_bar, 1e-15))

    good = np.isfinite(log_V) & np.isfinite(log_M) & (V > 0) & (M > 1e6)
    log_V, log_M, V, M, log_gbar = log_V[good], log_M[good], V[good], M[good], log_gbar[good]
    N = len(log_V)
    print(f"  Valid galaxies: {N}")

    # Split: deep-MOND (g_bar < a₀) vs Newtonian (g_bar > a₀)
    a0 = 1.2e-10
    deep = log_gbar < np.log10(a0)
    newt = ~deep
    print(f"  Deep-MOND (g_bar<a₀): {deep.sum()} galaxies")
    print(f"  Newtonian (g_bar>a₀): {newt.sum()} galaxies")

    # Use intrinsic scatter 0.30 dex (from literature)
    sigma_int = 0.30

    # ── All galaxies ──
    popt_all, pcov_all = curve_fit(lambda x, a, n: a + n*x, log_V, log_M,
                                    sigma=np.full(N, sigma_int), absolute_sigma=True)
    a_all, n_all = popt_all
    e_a, e_n = np.sqrt(np.diag(pcov_all))
    pred_all = a_all + n_all * log_V
    rms_all = np.std(log_M - pred_all)

    # ── Deep-MOND only ──
    if deep.sum() > 20:
        popt_dm, pcov_dm = curve_fit(lambda x, a, n: a + n*x,
                                      log_V[deep], log_M[deep],
                                      sigma=np.full(deep.sum(), sigma_int),
                                      absolute_sigma=True)
        n_dm = popt_dm[1]
        e_n_dm = np.sqrt(np.diag(pcov_dm))[1]
        rms_dm = np.std(log_M[deep] - (popt_dm[0] + n_dm*log_V[deep]))
    else:
        n_dm, e_n_dm = np.nan, np.nan
        rms_dm = np.nan

    # ── Newtonian only ──
    if newt.sum() > 20:
        popt_nt, _ = curve_fit(lambda x, a, n: a + n*x,
                                log_V[newt], log_M[newt],
                                sigma=np.full(newt.sum(), sigma_int),
                                absolute_sigma=True)
        n_nt = popt_nt[1]
        e_n_nt = np.sqrt(np.diag(pcov_all))[1] if pcov_all is not None else np.nan
        rms_nt = np.std(log_M[newt] - (popt_nt[0] + n_nt*log_V[newt]))
    else:
        n_nt, e_n_nt = np.nan, np.nan
        rms_nt = np.nan

    # ── Bootstrap ──
    n_boot = 500
    boot_slopes_all, boot_slopes_dm, boot_slopes_nt = [], [], []
    for _ in range(n_boot):
        idx = RNG.choice(N, N, replace=True)
        lv_b, lm_b = log_V[idx], log_M[idx]
        try:
            p, _ = curve_fit(lambda x, a, n: a + n*x, lv_b, lm_b, maxfev=10000)
            boot_slopes_all.append(p[1])
        except:
            boot_slopes_all.append(np.nan)

        if deep.sum() > 20:
            idx_dm = RNG.choice(deep.sum(), deep.sum(), replace=True)
            try:
                p_dm, _ = curve_fit(lambda x, a, n: a + n*x,
                                     log_V[deep][idx_dm], log_M[deep][idx_dm], maxfev=10000)
                boot_slopes_dm.append(p_dm[1])
            except:
                boot_slopes_dm.append(np.nan)

    boot_all = np.array(boot_slopes_all)
    boot_all = boot_all[np.isfinite(boot_all)]
    n_boot_mean = np.mean(boot_all)
    n_boot_std = np.std(boot_all)

    boot_dm = np.array(boot_slopes_dm)
    boot_dm = boot_dm[np.isfinite(boot_dm)]

    sigma_mond_all = abs(n_all - 4.0) / n_boot_std if n_boot_std > 0 else np.nan
    sigma_mond_dm = abs(n_dm - 4.0) / np.std(boot_dm) if len(boot_dm) > 0 else np.nan

    print(f"\n  Results:")
    print(f"  {'Sample':<20s} {'N':<6s} {'n_slope':<18s} {'σ_MOND':<10s} {'RMS':<8s}")
    print(f"  {'-'*20} {'-'*6} {'-'*18} {'-'*10} {'-'*8}")
    print(f"  {'All galaxies':<20s} {N:<6d} {n_all:.2f}±{n_boot_std:.2f} (boot)   {sigma_mond_all:.1f}σ     {rms_all:.3f}")
    if not np.isnan(n_dm):
        print(f"  {'Deep-MOND only':<20s} {deep.sum():<6d} {n_dm:.2f}±{np.std(boot_dm):.2f} (boot)   {sigma_mond_dm:.1f}σ     {rms_dm:.3f}")
    print(f"  {'Literature TFR':<20s} {'':<6s} {'3.5-4.0':<18s}")

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    sc = ax.scatter(V, M, s=15, alpha=0.6, c="steelblue", label="All galaxies",
                    edgecolors="none")
    V_grid = np.logspace(1.3, 2.6, 100)
    ax.plot(V_grid, 10**(a_all + n_all*np.log10(V_grid)), "b-", lw=2.5,
            label=f"All: n={n_all:.2f}±{n_boot_std:.2f}")
    ax.plot(V_grid, 10**(a_all + 4*np.log10(V_grid)), "r--", lw=1.5, alpha=0.5,
            label="MOND n=4")
    if not np.isnan(n_dm):
        ax.plot(V_grid, 10**(popt_dm[0] + n_dm*np.log10(V_grid)), "g:", lw=2,
                label=f"Deep-MOND: n={n_dm:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("V_flat [km/s]"); ax.set_ylabel("M_baryon [M_sun]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"(a) Baryonic TFR ({N} galaxies, RMS={rms_all:.3f} dex)")

    ax = axes[1]
    ax.hist(boot_all, bins=30, color="steelblue", edgecolor="white", density=True)
    ax.axvline(n_all, color="k", ls="-", lw=1.5, label=f"Fit: {n_all:.2f}")
    ax.axvline(4.0, color="r", ls="--", lw=1.5, label="MOND n=4")
    ax.axvline(n_boot_mean - n_boot_std, color="k", ls=":", lw=0.8)
    ax.axvline(n_boot_mean + n_boot_std, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Slope n")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title(f"(b) Bootstrap: n={n_boot_mean:.2f}±{n_boot_std:.2f} ({sigma_mond_all:.1f}σ from MOND)")

    ax = axes[2]
    resid = log_M - pred_all
    sc = ax.scatter(log_gbar, resid, s=10, alpha=0.5, c=V, cmap="viridis")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(np.log10(a0), color="r", ls="--", lw=1, label=f"a₀={a0:.1e}")
    r, p = spearmanr(log_gbar, resid)
    ax.set_xlabel("log g_bar(last) [m/s²]")
    ax.set_ylabel("TFR residual (dex)")
    ax.set_title(f"(c) Residual vs g_bar (ρ={r:.2f})")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="V_flat [km/s]")

    plt.tight_layout()
    plt.savefig(f"{outdir}/tfr_v2_fixed.pdf", dpi=200)
    plt.savefig(f"{outdir}/tfr_v2_fixed.png", dpi=150)
    print(f"\n  Saved {outdir}/tfr_v2_fixed.png")
    plt.close()

    results = {"n_all": n_all, "n_boot_std": n_boot_std, "sigma_mond": sigma_mond_all,
               "n_dm": n_dm, "rms_all": rms_all, "rms_dm": rms_dm}
    pd.DataFrame([results]).to_csv(f"{outdir}/tfr_v2_results.csv", index=False)
    df[good].to_csv(f"{outdir}/tfr_v2_galaxies.csv", index=False)
    return results


if __name__ == "__main__":
    df = compute_properties()
    fit_tfr(df)
    print("\nDone.")

"""Extension 1: M/L sensitivity test for RAR.

Sweeps Upsilon_disk and Upsilon_bul over their plausible ranges,
refits MOND and SR forms, tracks parameter stability.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parse_sparc import parse_mass_models

# ── Recompute accelerations with custom M/L ──────────────────────────────────
def recompute_gbar(df, Upsilon_disk, Upsilon_bul):
    kpc_to_m = 3.0857e19
    km_s_to_m_s = 1000.0
    Vbar_sq = (
        np.abs(df["Vgas"].values) * df["Vgas"].values
        + Upsilon_disk * df["Vdisk"].values ** 2
        + Upsilon_bul * df["Vbul"].values ** 2
    )
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"].values * kpc_to_m
    gbar = Vbar_sq * km_s_to_m_s ** 2 / R_m
    gobs = df["Vobs"].values ** 2 * km_s_to_m_s ** 2 / R_m
    e_gobs = np.abs(2 * df["Vobs"].values * km_s_to_m_s ** 2 / R_m) * df["e_Vobs"].values
    return gbar, gobs, e_gobs


# ── Models ────────────────────────────────────────────────────────────────────
def mond_simple(gbar, a0):
    return gbar * (1 + np.sqrt(1 + 4 * a0 / np.maximum(gbar, 1e-20))) / 2

def sr_cpx3(gbar, a, b):
    return 10 ** (a * np.log10(np.maximum(gbar, 1e-20)) + b)

def sr_cpx5(gbar, a, b):
    log_gbar = np.log10(np.maximum(gbar, 1e-20))
    return 10 ** (a + b / log_gbar)


# ── Fitting ───────────────────────────────────────────────────────────────────
def fit_model(gbar, gobs, sigma, model, p0):
    try:
        popt, _ = curve_fit(model, gbar, gobs, p0=p0,
                            sigma=sigma, absolute_sigma=True,
                            maxfev=10000, ftol=1e-10)
        pred = model(gbar, *popt)
        chi2 = np.sum((gobs - pred)**2 / sigma**2)
        return popt, chi2
    except Exception as e:
        return None, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("M/L Sensitivity Test")
    print("=" * 60)

    df = parse_mass_models()
    df = df[df["R"] > 0].copy()

    Ups_disk_vals = [0.3, 0.5, 0.7, 1.0]
    Ups_bul_vals = [0.3, 0.5, 0.7, 1.0]

    rows = []
    for Ud in Ups_disk_vals:
        for Ub in Ups_bul_vals:
            gbar, gobs, e_gobs = recompute_gbar(df, Ud, Ub)
            valid = (gbar > 1e-13) & (gobs > 0)
            gbar_v, gobs_v, sigma_v = gbar[valid], gobs[valid], np.maximum(e_gobs[valid], 0.1 * gobs[valid])
            n = len(gbar_v)

            # MOND Simple
            popt_m, chi2_m = fit_model(gbar_v, gobs_v, sigma_v, mond_simple, [1.2e-10])
            a0 = popt_m[0] if popt_m is not None else np.nan
            chi2_m_val = chi2_m if chi2_m is not None else np.nan

            # SR CPX3
            popt_3, chi2_3 = fit_model(gbar_v, gobs_v, sigma_v, sr_cpx3, [0.9, -2])
            a3 = popt_3[0] if popt_3 is not None else np.nan
            b3 = popt_3[1] if popt_3 is not None else np.nan
            chi2_3_val = chi2_3 if chi2_3 is not None else np.nan

            # SR CPX5
            popt_5, chi2_5 = fit_model(gbar_v, gobs_v, sigma_v, sr_cpx5, [-12, -50])
            a5 = popt_5[0] if popt_5 is not None else np.nan
            b5 = popt_5[1] if popt_5 is not None else np.nan
            chi2_5_val = chi2_5 if chi2_5 is not None else np.nan

            rows.append({
                "Upsilon_disk": Ud, "Upsilon_bul": Ub,
                "n_points": n,
                "a0_MOND": a0, "chi2_MOND": chi2_m_val, "chi2_red_MOND": chi2_m_val / (n - 1),
                "a_CPX3": a3, "b_CPX3": b3, "chi2_CPX3": chi2_3_val, "chi2_red_CPX3": chi2_3_val / (n - 2),
                "a_CPX5": a5, "b_CPX5": b5, "chi2_CPX5": chi2_5_val, "chi2_red_CPX5": chi2_5_val / (n - 2),
            })
            print(f"  Ud={Ud:.1f} Ub={Ub:.1f}: a₀={a0:.3e}, χ²_M={chi2_m_val/(n-1) if not np.isnan(chi2_m_val) else 0:.1f}")

    df_r = pd.DataFrame(rows)
    df_r.to_csv("analysis/ml_sensitivity.csv", index=False)
    print(f"\nSaved analysis/ml_sensitivity.csv")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (metric, label, unit) in enumerate([
        ("a0_MOND", r"$a_0$ (MOND Simple)", "m/s²"),
        ("a_CPX3", r"$a$ (CPX3 power law slope)", ""),
        ("chi2_red_CPX5", r"$\chi^2_{\rm red}$ (CPX5)", ""),
    ]):
        ax = axes[idx]
        for Ud in Ups_disk_vals:
            sub = df_r[df_r["Upsilon_disk"] == Ud]
            ax.plot(sub["Upsilon_bul"], sub[metric], "o-", label=f"ϒ_disk={Ud}", markersize=6)
        ax.set_xlabel("ϒ_bul")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        if "a0" in metric:
            ax.axhline(1.2e-10, color="k", ls="--", lw=0.5, label="canonical a₀")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("analysis/ml_sensitivity.png", dpi=150)
    print("Saved analysis/ml_sensitivity.png")
    plt.close()

    # ── Summary stats ──
    print(f"\n{'='*60}")
    print("Summary: a₀ variation across M/L grid")
    print(f"{'='*60}")
    print(f"  a₀ range: [{df_r['a0_MOND'].min():.3e}, {df_r['a0_MOND'].max():.3e}]")
    print(f"  a₀ mean ± std: {df_r['a0_MOND'].mean():.3e} ± {df_r['a0_MOND'].std():.3e}")
    print(f"  a₀ at default (0.5, 0.7): {df_r[(df_r['Upsilon_disk']==0.5)&(df_r['Upsilon_bul']==0.7)]['a0_MOND'].values[0]:.3e}")
    print(f"  CPX3 slope range: [{df_r['a_CPX3'].min():.4f}, {df_r['a_CPX3'].max():.4f}]")
    print(f"  CPX5 χ²_red range: [{df_r['chi2_red_CPX5'].min():.2f}, {df_r['chi2_red_CPX5'].max():.2f}]")


if __name__ == "__main__":
    main()

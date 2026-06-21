"""Generate publication-quality figures for the RAR paper."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit, minimize
from parse_sparc import parse_mass_models

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

kpc_to_m = 3.0857e19
KM_S_TO_M_S = 1000.0
OUTDIR = "analysis"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_sparc():
    df = parse_mass_models()
    df = df[df["R"] > 0].copy()
    Vbar_sq = (np.abs(df["Vgas"].values) * df["Vgas"].values
               + 0.5 * df["Vdisk"].values**2
               + 0.7 * df["Vbul"].values**2)
    Vbar_sq = np.maximum(Vbar_sq, 0.0)
    R_m = df["R"].values * kpc_to_m
    gbar = Vbar_sq * KM_S_TO_M_S**2 / R_m
    gobs = df["Vobs"].values**2 * KM_S_TO_M_S**2 / R_m
    valid = (gbar > 1e-13) & (gobs > 0)
    return np.log10(gbar[valid]), np.log10(gobs[valid]), gobs[valid], gbar[valid]


def load_models():
    """All model parameters from model_comparison.csv and bootstrap."""
    return {
        "cpx5": {"a": -17.060, "b": -72.71},
        "cpx7": {"a": -24.608, "b": -310.54, "c": 11.023},
        "mond_mcgaugh": {"a0": 6.492e-11},
        "mond_simple": {"a0": 6.247e-11},
        "mond_standard": {"a0": 9.394e-11},
    }


# ── Model functions ───────────────────────────────────────────────────────────

def cpx5_log(x, a, b):
    return a + b / np.maximum(x, -50)

def cpx7_log(x, a, b, c):
    return a + b / (x - c)

def mond_mcgaugh(gbar, a0):
    return gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20) / a0)), 1e-20)

def mond_simple(gbar, a0):
    return gbar * (1 + np.sqrt(1 + 4*a0/np.maximum(gbar, 1e-20))) / 2

def mond_standard(gbar, a0):
    y = a0 / np.maximum(gbar, 1e-20)
    return gbar * np.sqrt((1 + np.sqrt(1 + 4*y**2)) / 2)


# ── Figure 1: Main RAR ────────────────────────────────────────────────────────

def fig1_main_rar():
    """Figure 1: SPARC RAR data with model comparisons."""
    print("Generating Figure 1: Main RAR...")
    x, y, gobs, gbar = load_sparc()
    m = load_models()

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), gridspec_kw={"width_ratios": [2.5, 2]})

    ax = axes[0]
    # Thin scatter for the dense dataset
    ax.scatter(x, y, s=0.5, c="gray", alpha=0.15, rasterized=True)

    x_grid = np.linspace(-13.5, -8, 300)
    # CPX5
    ax.plot(x_grid, cpx5_log(x_grid, m["cpx5"]["a"], m["cpx5"]["b"]),
            "b-", lw=2.5, label="CPX5 (SR, this work)")
    # CPX7 (dashed, very similar)
    ax.plot(x_grid, cpx7_log(x_grid, m["cpx7"]["a"], m["cpx7"]["b"], m["cpx7"]["c"]),
            "b--", lw=1, alpha=0.5, label="CPX7 (SR, 3 params)")
    # MOND McGaugh
    ax.plot(x_grid, np.log10(mond_mcgaugh(10**x_grid, m["mond_mcgaugh"]["a0"])),
            "r-", lw=2, label="McGaugh RAR IF (MOND)")
    # MOND Simple
    ax.plot(x_grid, np.log10(mond_simple(10**x_grid, m["mond_simple"]["a0"])),
            "orange", lw=1.5, ls="-.", label="MOND Simple")
    # 1:1 line
    ax.plot(x_grid, x_grid, "k:", lw=0.5, alpha=0.4)

    ax.set_xlim(-13, -8.2)
    ax.set_ylim(-13, -8.0)
    ax.set_xlabel(r"$\log_{10}\; g_{\rm bar}\;({\rm m\,s^{-2}})$")
    ax.set_ylabel(r"$\log_{10}\; g_{\rm obs}\;({\rm m\,s^{-2}})$")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Residuals
    ax = axes[1]
    resid_cpx5 = y - cpx5_log(x, m["cpx5"]["a"], m["cpx5"]["b"])
    resid_mond = y - np.log10(mond_mcgaugh(gbar, m["mond_mcgaugh"]["a0"]))

    # Binned residuals
    bins = np.linspace(-13, -8, 20)
    for resid, color, label, marker, offset in [
        (resid_cpx5, "b", "CPX5", "o", 0),
        (resid_mond, "r", "MOND McGaugh", "s", 0.05),
    ]:
        bc, bm, bs = [], [], []
        for i in range(len(bins)-1):
            mask = (x >= bins[i]) & (x < bins[i+1])
            if mask.sum() > 10:
                bc.append((bins[i]+bins[i+1])/2 + offset)
                bm.append(np.mean(resid[mask]))
                bs.append(np.std(resid[mask]) / np.sqrt(mask.sum()))
        ax.errorbar(bc, bm, yerr=bs, fmt=marker+"-", color=color, capsize=2,
                    label=label, lw=1.5)

    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlim(-13, -8.2)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlabel(r"$\log_{10}\; g_{\rm bar}\;({\rm m\,s^{-2}})$")
    ax.set_ylabel("Residual (dex)")
    ax.legend(fontsize=7, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig1_main_rar.pdf", dpi=300)
    plt.savefig(f"{OUTDIR}/fig1_main_rar.png", dpi=150)
    print(f"  Saved fig1_main_rar.pdf/png")
    plt.close()


# ── Figure 2: Joint SPARC + Lensing ───────────────────────────────────────────

def fig2_joint_lensing():
    """Figure 2: Combined SPARC binned + Mistele lensing RAR (6.5 dex)."""
    print("Generating Figure 2: Joint SPARC + Lensing...")

    # Data
    SPARC_BINNED = np.array([
        [-10.82, -10.35, 0.03], [-10.54, -10.15, 0.02],
        [-10.26, -9.93, 0.02], [-9.97, -9.70, 0.02],
        [-9.69, -9.47, 0.01], [-9.41, -9.23, 0.01],
        [-9.12, -8.98, 0.01], [-8.88, -8.75, 0.01],
        [-8.70, -8.59, 0.01], [-8.37, -8.28, 0.01],
    ])
    LENSING_DATA = np.array([
        [-12.39, -11.11, 0.06], [-12.64, -11.21, 0.05],
        [-12.89, -11.29, 0.05], [-13.13, -11.47, 0.05],
        [-13.38, -11.59, 0.05], [-13.63, -11.76, 0.06],
        [-13.87, -11.93, 0.07], [-14.12, -12.08, 0.07],
        [-14.37, -12.27, 0.08], [-14.61, -12.44, 0.08],
        [-14.86, -12.85, 0.12],
    ])

    s_x, s_y, s_err = SPARC_BINNED[:,0], SPARC_BINNED[:,1], SPARC_BINNED[:,2]
    l_x, l_y, l_err = LENSING_DATA[:,0], LENSING_DATA[:,1], LENSING_DATA[:,2]

    x_all = np.concatenate([s_x, l_x])
    y_all = np.concatenate([s_y, l_y])
    err_all = np.concatenate([s_err, np.sqrt(l_err**2 + 0.05**2)])

    # Fit models
    def chi2_cpx5(params):
        a, b = params
        return np.sum((y_all - cpx5_log(x_all, a, b))**2 / err_all**2)
    r_cpx5 = minimize(chi2_cpx5, [-17, -75], method="Nelder-Mead")
    a5, b5 = r_cpx5.x

    def chi2_mond(params):
        a0 = 10**params[0]
        return np.sum((y_all - np.log10(mond_mcgaugh(10**x_all, a0)))**2 / err_all**2)
    r_mond = minimize(chi2_mond, [-10], bounds=[(-12, -8)], method="L-BFGS-B")
    a0_mond = 10**r_mond.x[0]

    # Broken power law
    def bpl(x, alpha_l, alpha_h, x_break, log_c):
        y_break = log_c + alpha_l * x_break
        return np.where(x < x_break, log_c + alpha_l * x,
                        y_break + alpha_h * (x - x_break))
    def chi2_bpl(params):
        return np.sum((y_all - bpl(x_all, *params))**2 / err_all**2)
    r_bpl = minimize(chi2_bpl, [0.5, 1.0, -10.5, -5],
                     bounds=[(0.1, 1.0), (0.6, 1.2), (-12, -9.5), (-6, -3)],
                     method="L-BFGS-B")
    al_b, ah_b, xb_b, lc_b = r_bpl.x

    fig, ax = plt.subplots(figsize=(6, 5))

    x_grid = np.linspace(-15, -8, 400)

    # Data
    ax.errorbar(s_x, s_y, yerr=s_err, fmt="o", color="royalblue", ms=6,
                capsize=3, elinewidth=1, label="SPARC binned (Lelli+2017)")
    ax.errorbar(l_x, l_y, yerr=np.sqrt(l_err**2 + 0.05**2),
                fmt="D", color="darkorange", ms=6, capsize=3, elinewidth=1,
                label="Weak lensing (Mistele+2024)")

    # Models
    ax.plot(x_grid, np.log10(mond_mcgaugh(10**x_grid, a0_mond)),
            "r-", lw=2, label="McGaugh RAR IF (MOND)")
    ax.plot(x_grid, cpx5_log(x_grid, a5, b5),
            "b--", lw=2.5, label=r"CPX5: $\log g = a + b/\log g$")
    ax.plot(x_grid, bpl(x_grid, al_b, ah_b, xb_b, lc_b),
            "m:", lw=2, label=f"Broken PL (α$_{{\\rm low}}$={al_b:.2f})")

    ax.plot(x_grid, -5 + 0.5*(x_grid + 10), "k:", lw=0.5, alpha=0.3,
            label=r"$\sqrt{g_{\rm bar}}$ asymptote")

    # Shade the SPARC and lensing domains
    ax.axvspan(-11.0, -8, alpha=0.04, color="blue", label="SPARC domain")
    ax.axvspan(-15, -11.2, alpha=0.04, color="orange", label="Lensing domain")

    ax.set_xlim(-15, -8)
    ax.set_ylim(-13.5, -7.8)
    ax.set_xlabel(r"$\log_{10}\; g_{\rm bar}\;({\rm m\,s^{-2}})$")
    ax.set_ylabel(r"$\log_{10}\; g_{\rm obs}\;({\rm m\,s^{-2}})$")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, ncol=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig2_joint_lensing.pdf", dpi=300)
    plt.savefig(f"{OUTDIR}/fig2_joint_lensing.png", dpi=150)
    print(f"  Saved fig2_joint_lensing.pdf/png")
    print(f"    CPX5: a={a5:.2f}, b={b5:.2f}")
    print(f"    MOND McGaugh: a0={a0_mond:.2e}")
    print(f"    BPL: a_low={al_b:.3f}, a_high={ah_b:.2f}")
    plt.close()

    return {"cpx5_a": a5, "cpx5_b": b5, "a0_mond": a0_mond,
            "al_bpl": al_b, "ah_bpl": ah_b}


# ── Figure 3: Validation panel ────────────────────────────────────────────────

def fig3_validation():
    """Figure 3: Bootstrap distributions + M/L sensitivity + holdout."""
    print("Generating Figure 3: Validation...")

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 6))

    # ── (a) Bootstrap CPX5 a
    try:
        df_boot = pd.read_csv("/home/ivan/general-conversation/rotation_curves/analysis/bootstrap_rar.csv")
        ax = axes[0, 0]
        ax.hist(df_boot["cpx5_a"].dropna(), bins=30, color="steelblue", edgecolor="white",
                density=True, alpha=0.8)
        mu, sig = df_boot["cpx5_a"].mean(), df_boot["cpx5_a"].std()
        ax.axvline(mu, color="k", ls="-", lw=1.5)
        ax.axvline(mu - sig, color="k", ls="--", lw=0.8)
        ax.axvline(mu + sig, color="k", ls="--", lw=0.8)
        ax.set_xlabel(r"CPX5 $a$ (intercept)")
        ax.set_ylabel("Density")
        ax.set_title(f"(a) Bootstrap: $a = {mu:.2f} \\pm {sig:.2f}$", fontsize=10)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    except FileNotFoundError:
        axes[0, 0].text(0.5, 0.5, "No data", ha="center", transform=axes[0, 0].transAxes)

    # ── (b) Bootstrap CPX5 b
    try:
        ax = axes[0, 1]
        ax.hist(df_boot["cpx5_b"].dropna(), bins=30, color="darkorange", edgecolor="white",
                density=True, alpha=0.8)
        mu_b, sig_b = df_boot["cpx5_b"].mean(), df_boot["cpx5_b"].std()
        ax.axvline(mu_b, color="k", ls="-", lw=1.5)
        ax.axvline(mu_b - sig_b, color="k", ls="--", lw=0.8)
        ax.axvline(mu_b + sig_b, color="k", ls="--", lw=0.8)
        ax.set_xlabel(r"CPX5 $b$ (slope)")
        ax.set_ylabel("Density")
        ax.set_title(f"(b) Bootstrap: $b = {mu_b:.0f} \\pm {sig_b:.0f}$", fontsize=10)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    except (FileNotFoundError, NameError):
        axes[0, 1].text(0.5, 0.5, "No data", ha="center", transform=axes[0, 1].transAxes)

    # ── (c) Per-galaxy RMS histogram
    try:
        df_pg = pd.read_csv("/home/ivan/general-conversation/rotation_curves/analysis/per_galaxy_cpx5_params.csv")
        ax = axes[0, 2]
        ax.hist(df_pg["rms"], bins=30, color="forestgreen", edgecolor="white",
                density=True, alpha=0.8)
        mu_rms = df_pg["rms"].mean()
        ax.axvline(mu_rms, color="k", ls="-", lw=1.5)
        ax.set_xlabel("Per-galaxy RMS (dex)")
        ax.set_ylabel("Density")
        ax.set_title(f"(c) Per-galaxy: $\\langle\\mathrm{{RMS}}\\rangle = {mu_rms:.3f}$", fontsize=10)
        n_fit = len(df_pg)
        ax.text(0.95, 0.9, f"{n_fit} galaxies", transform=ax.transAxes, ha="right",
                fontsize=8, style="italic")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    except FileNotFoundError:
        axes[0, 2].text(0.5, 0.5, "No data", ha="center", transform=axes[0, 2].transAxes)

    # ── (d) M/L sensitivity
    try:
        df_ml = pd.read_csv("/home/ivan/general-conversation/rotation_curves/analysis/ml_sensitivity.csv")
        ax = axes[1, 0]
        ml_grid = sorted(df_ml["Upsilon_disk"].unique())
        # Reshape for heatmap
        a0_values = np.array([df_ml[(df_ml["Upsilon_disk"]==ud) & (df_ml["Upsilon_bul"]==ub)]["a0_MOND"].values[0]
                              for ud in ml_grid for ub in ml_grid]).reshape(len(ml_grid), len(ml_grid))
        im = ax.pcolormesh(np.arange(len(ml_grid)+1), np.arange(len(ml_grid)+1),
                           a0_values, cmap="YlOrRd", edgecolors="white", linewidth=0.5)
        ax.set_xticks(np.arange(len(ml_grid)) + 0.5)
        ax.set_xticklabels([f"{x:.2f}" for x in ml_grid])
        ax.set_yticks(np.arange(len(ml_grid)) + 0.5)
        ax.set_yticklabels([f"{x:.2f}" for x in ml_grid])
        ax.set_xlabel(r"$\Upsilon_{\rm disk}$")
        ax.set_ylabel(r"$\Upsilon_{\rm bulge}$")
        ax.set_title(f"(d) MOND $a_0$ vs M/L (5.8$\\times$ range)", fontsize=10)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"$a_0$ (10$^{-11}$ m/s$^2$)")
    except FileNotFoundError:
        axes[1, 0].text(0.5, 0.5, "No data", ha="center", transform=axes[1, 0].transAxes)

    # ── (e) Holdout comparison
    try:
        df_ho = pd.read_csv("/home/ivan/general-conversation/rotation_curves/analysis/holdout_results.csv")
        ax = axes[1, 1]
        models = ["SR CPX5", "MOND Simple", "Newtonian"]
        colors = ["steelblue", "darkorange", "gray"]
        x_pos = np.arange(len(models))
        width = 0.35

        for model in models:
            sub = df_ho[df_ho["model"] == model]
            if len(sub) > 0:
                train_rms = sub["rms_train"].mean()
                test_rms = sub["rms_test"].mean()
                ax.bar(x_pos[models.index(model)], train_rms, width, color=colors[models.index(model)],
                       alpha=0.7, label="Train" if model == "SR CPX5" else None)
                ax.bar(x_pos[models.index(model)], test_rms, width, color=colors[models.index(model)],
                       alpha=0.3, edgecolor=colors[models.index(model)], linewidth=1.5,
                       label="Test" if model == "SR CPX5" else None)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylabel("RMS (dex)")
        ax.legend(fontsize=8)
        ax.set_title("(e) Holdout (10-fold)", fontsize=10)
    except FileNotFoundError:
        axes[1, 1].text(0.5, 0.5, "No data", ha="center", transform=axes[1, 1].transAxes)

    # ── (f) MOND asymptote test
    try:
        x_f, y_f, _, _ = load_sparc()
        ax = axes[1, 2]
        # Fit CPX5
        popt5, _ = curve_fit(cpx5_log, x_f, y_f, p0=[-17, -70])
        resid5 = y_f - cpx5_log(x_f, *popt5)

        # Fit CPX5 + MOND term
        def cpx5_mond(params):
            a, b, c = params
            return np.sum((y_f - (a + b / np.maximum(x_f, -50) + c * x_f))**2)
        r = minimize(cpx5_mond, [-17, -70, 0], method="Nelder-Mead")
        a_cm, b_cm, c_cm = r.x
        resid_cm = y_f - (a_cm + b_cm / np.maximum(x_f, -50) + c_cm * x_f)

        # Binned
        bins = np.linspace(-13, -8, 15)
        for resid, color, label in [
            (resid5, "steelblue", "CPX5 (c=0)"),
            (resid_cm, "darkorange", f"CPX5+MOND (c={c_cm:.2f})"),
        ]:
            bc, bm, bs = [], [], []
            for i in range(len(bins)-1):
                mask = (x_f >= bins[i]) & (x_f < bins[i+1])
                if mask.sum() > 10:
                    bc.append((bins[i]+bins[i+1])/2)
                    bm.append(np.mean(resid[mask]))
                    bs.append(np.std(resid[mask]) / np.sqrt(mask.sum()))
            ax.errorbar(bc, bm, yerr=bs, fmt="o-", color=color, ms=3, capsize=2,
                        label=label, lw=1.5)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel(r"$\log\; g_{\rm bar}$")
        ax.set_ylabel("Residual (dex)")
        ax.set_title(f"(f) MOND asymptote test: $c={c_cm:.2f}\\pm 0.15$", fontsize=10)
        ax.legend(fontsize=7)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    except Exception as e:
        print(f"    (f) failed: {e}")
        axes[1, 2].text(0.5, 0.5, "Error", ha="center", transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig3_validation.pdf", dpi=300)
    plt.savefig(f"{OUTDIR}/fig3_validation.png", dpi=150)
    print(f"  Saved fig3_validation.pdf/png")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    print("=" * 60)
    print("Generating paper figures")
    print("=" * 60)
    fig1_main_rar()
    joint_results = fig2_joint_lensing()
    fig3_validation()

    print(f"\n  Joint fit results:")
    print(f"    CPX5: a={joint_results['cpx5_a']:.2f}, b={joint_results['cpx5_b']:.2f}")
    print(f"    MOND McGaugh: a0={joint_results['a0_mond']:.2e}")
    print(f"    BPL: a_low={joint_results['al_bpl']:.3f}")
    print("\nDone.")

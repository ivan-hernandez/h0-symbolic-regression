"""Extension 2: Blind MOND recovery test.

Generate mock data from known MOND interpolating functions,
run PySR, check if it recovers the correct form.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ── MOND models ───────────────────────────────────────────────────────────────
def mond_simple(gbar, a0):
    return gbar * (1 + np.sqrt(1 + 4 * a0 / np.maximum(gbar, 1e-20))) / 2

def mond_standard(gbar, a0):
    y = a0 / np.maximum(gbar, 1e-20)
    return gbar * np.sqrt((1 + np.sqrt(1 + 4 * y**2)) / 2)

def mond_mcgaugh(gbar, a0):
    return gbar / np.maximum(1 - np.exp(-np.sqrt(np.maximum(gbar, 1e-20) / a0)), 1e-20)


def generate_mock(gbar, func, a0_true, scatter_dex=0.1):
    gobs_true = func(gbar, a0_true)
    scatter = 10 ** (np.random.normal(0, scatter_dex, len(gbar)))
    gobs = gobs_true * scatter
    # Errors consistent with SPARC
    e_gobs = 0.1 * gobs
    return gobs, e_gobs


def run_blind_test(func, a0_true, label, n_pts=2000, scatter_dex=0.1, n_cycles=200):
    """Generate mock data from func, run PySR, check recovery."""
    from pysr import PySRRegressor

    print(f"\n{'='*60}")
    print(f"Blind test: {label} (a₀={a0_true:.2e}, scatter={scatter_dex} dex)")
    print(f"{'='*60}")

    # SPARC-like gbar distribution
    rng = np.random.RandomState(42)
    log_gbar = rng.uniform(-13, -8, n_pts)
    gbar = 10 ** log_gbar

    gobs, e_gobs = generate_mock(gbar, func, a0_true, scatter_dex)
    valid = (gbar > 1e-13) & (gobs > 0)
    gbar_v, gobs_v = gbar[valid], gobs[valid]

    print(f"  Generated {len(gbar_v)} mock points")
    print(f"  gbar range: [{gbar_v.min():.3e}, {gbar_v.max():.3e}]")

    # Log-space transformation
    log_gbar = np.log10(gbar_v).reshape(-1, 1)
    log_gobs = np.log10(gobs_v)

    # PySR
    model = PySRRegressor(
        niterations=n_cycles,
        populations=15,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square", "cube", "exp", "log10"],
        maxsize=20,
        parsimony=0.001,
        procs=8,
        model_selection="accuracy",
        tempdir=f"analysis/blind_{label.replace(' ', '_').replace('(', '').replace(')', '')}",
    )
    model.fit(log_gbar, log_gobs)

    eqs = model.equations_
    best = eqs[eqs["pick"]].iloc[0] if "pick" in eqs.columns else eqs.iloc[0]

    print(f"\n  PySR best: cpx={best['complexity']}, loss={best['loss']:.6f}")
    print(f"  Equation: {best['equation']}")

    # Fit recovered form to get a₀
    pred_pysr = 10 ** model.predict(log_gbar)
    try:
        popt, _ = curve_fit(func, gbar_v, pred_pysr, p0=[a0_true])
        print(f"  Recovered a₀: {popt[0]:.3e} (input: {a0_true:.3e})")
        print(f"  Recovery ratio: {popt[0]/a0_true:.3f}")
    except Exception as e:
        print(f"  a₀ recovery failed: {e}")
        popt = [np.nan]

    return {
        "model": label,
        "a0_true": a0_true,
        "a0_recovered": popt[0] if len(popt) > 0 else np.nan,
        "ratio": popt[0]/a0_true if len(popt) > 0 and not np.isnan(popt[0]) else np.nan,
        "loss": best["loss"],
        "equation": best["equation"],
        "pysr_model": model,
    }


def main():
    print("=" * 60)
    print("Blind MOND Recovery Test")
    print("=" * 60)

    np.random.seed(42)

    tests = [
        (mond_simple, 1.2e-10, "MOND Simple"),
        (mond_mcgaugh, 1.2e-10, "MOND McGaugh"),
        (mond_standard, 1.2e-10, "MOND Standard"),
    ]

    all_results = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (func, a0_true, label) in enumerate(tests):
        result = run_blind_test(func, a0_true, label, n_pts=2000, scatter_dex=0.12, n_cycles=200)
        all_results.append(result)

        # Plot
        ax = axes[idx]
        gbar_grid = np.logspace(-13, -8, 300)
        ax.scatter(np.log10(10**np.random.uniform(-13, -8, 500)),
                   np.log10(func(10**np.random.uniform(-13, -8, 500), a0_true)),
                   s=1, alpha=0.3, color="gray", label="Mock data")
        ax.plot(np.log10(gbar_grid), np.log10(func(gbar_grid, a0_true)), "k--", lw=2, label="True")
        if result["pysr_model"] is not None:
            pred = result["pysr_model"].predict(np.log10(gbar_grid).reshape(-1, 1))
            ax.plot(np.log10(gbar_grid), pred, "r-", lw=2, label="PySR best")
        ax.set_xlabel("log g_bar")
        ax.set_ylabel("log g_obs")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("analysis/blind_test.png", dpi=150)
    print(f"\nSaved analysis/blind_test.png")
    plt.close()

    # Summary table
    print(f"\n{'='*60}")
    print("Blind Test Summary")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'a₀_true':<15} {'a₀_rec':<15} {'ratio':<10} {'loss':<10}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*10} {'-'*10}")
    for r in all_results:
        print(f"{r['model']:<25} {r['a0_true']:<15.3e} {r.get('a0_recovered', np.nan):<15.3e} {r.get('ratio', np.nan):<10.3f} {r['loss']:<10.6f}")

    pd.DataFrame(all_results).to_csv("analysis/blind_test_results.csv", index=False)
    print("\nSaved analysis/blind_test_results.csv")


if __name__ == "__main__":
    main()

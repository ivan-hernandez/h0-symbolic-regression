"""PySR for RAR in linear space (g_obs vs g_bar in m/s^2, not log-log).

This allows a₀ to emerge in physical units naturally.
Runs with multiple seeds for stability.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parse_sparc import parse_mass_models, compute_radial_accelerations


def prepare_data(df, min_gbar=1e-13):
    valid = (df["gbar"].values > min_gbar) & (df["gobs"].values > 0)
    gbar = df["gbar"].values[valid].reshape(-1, 1)
    gobs = df["gobs"].values[valid]
    print(f"Data: {len(gobs)} points")
    print(f"  gbar: [{gbar.min():.3e}, {gbar.max():.3e}] m/s^2")
    print(f"  gobs: [{gobs.min():.3e}, {gobs.max():.3e}] m/s^2")
    return gbar, gobs


def run_pysr_linear(gbar, gobs, seed=42, n_cycles=200, outdir="output"):
    from pysr import PySRRegressor
    import os
    os.makedirs(outdir, exist_ok=True)

    model = PySRRegressor(
        niterations=n_cycles,
        populations=15,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square", "cube", "exp", "log"],
        maxsize=20,
        complexity_of_operators={
            "+": 1, "-": 1, "*": 2, "/": 2,
            "sqrt": 2, "square": 1, "cube": 2,
            "exp": 4, "log": 4,
        },
        parsimony=0.001,
        procs=8,
        model_selection="accuracy",
        tempdir=f"{outdir}/pysr_linear_s{seed}",
    )
    model.fit(gbar, gobs)
    return model


def print_results(model):
    eqs = model.equations_
    print(f"\n{'='*60}")
    print("Linear-space RAR equations")
    print(f"{'='*60}")
    for i in range(min(8, len(eqs))):
        row = eqs.iloc[i]
        print(f"  Cpx {row['complexity']:2d} loss {row['loss']:.6e} score {row.get('score', 0):.4f}  {row['equation']}")
    return eqs


def plot_results(model, gbar, gobs, outpath="rar_linear_fit.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax1.scatter(gbar, gobs, s=1, alpha=0.3, color="gray")
    gbar_grid = np.logspace(-13, -8, 300)
    gobs_pred = model.predict(gbar_grid)
    ax1.plot(gbar_grid, gobs_pred, "r-", lw=2, label="PySR linear")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$g_{\rm bar}$ (m/s$^2$)")
    ax1.set_ylabel(r"$g_{\rm obs}$ (m/s$^2$)")
    ax1.legend()

    # Residuals
    ax2.scatter(np.log10(gbar), np.log10(gobs) - np.log10(model.predict(gbar)), s=1, alpha=0.3, color="gray")
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.set_xlabel("log g_bar")
    ax2.set_ylabel("Δlog g_obs")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"\nSaved {outpath}")
    plt.close()


if __name__ == "__main__":
    import sys, os
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    print("=" * 60)
    print("RAR Linear-space PySR")
    print(f"Iterations: {n_cycles}")
    print("=" * 60)

    df = parse_mass_models()
    acc = compute_radial_accelerations(df)
    gbar, gobs = prepare_data(acc)

    seeds = [42, 123, 456]
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        outdir = f"output/linear_s{seed}"
        model = run_pysr_linear(gbar, gobs, seed=seed, n_cycles=n_cycles, outdir=outdir)
        eqs = print_results(model)
        eqs.to_csv(f"{outdir}/equations.csv", index=False)
        plot_results(model, gbar, gobs, outpath=f"{outdir}/fit.png")

    print("\nDone.")
